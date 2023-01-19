import copy
import math
import os
import shutil
from glob import glob
from typing import Iterable

import datasets
import diffusers
import hydra
import torch
import transformers
import wandb
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import StableDiffusionPipeline, DDPMScheduler, get_scheduler, DPMSolverMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import DictConfig, OmegaConf
import rich.tree
import rich.syntax
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from torchvision import transforms
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)


def print_config(cfg: DictConfig):
    style = "bright"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    fields = cfg.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_section = cfg.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=True)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)


def debug(port):
    logger.info("Connecting to debugger...")
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=port, stdoutToServer=True, stderrToServer=True)


def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def pixel_values_to_pil_images(pixel_values):
    images = (pixel_values / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    images = numpy_to_pil(images)
    return images


def get_allocated_cuda_memory():
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, 2)


class Model(torch.nn.Module):

    def __init__(
            self,
            unet,
            text_encoder,
            vae,
            unconditional_input_ids,
            classifier_free_ratio,
            noise_scheduler,
            use_pixel_loss
    ):
        super().__init__()
        self.unet = unet
        self.text_encoder = text_encoder
        self.vae = vae
        self.unconditional_input_ids = unconditional_input_ids
        self.classifier_free_ratio = classifier_free_ratio
        self.noise_scheduler = noise_scheduler
        self.use_pixel_loss = use_pixel_loss

    def get_image_pred(self, model_output, timestep, sample):

        t = timestep

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)

        # 1. compute alphas, betas
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t][:, None, None, None]
        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.noise_scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.noise_scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.noise_scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip "predicted x_0"
        if self.noise_scheduler.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        return pred_original_sample

    @torch.no_grad()
    def encode_image(self, image):
        latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * 0.18215
        return latents

    def decode_latents(self, latents):
        latents = (1 / 0.18215) * latents
        image = self.vae.decode(latents).sample
        return image

    def forward(self, pixel_values, input_ids):

        # classifier free
        classifier_free_mask = torch.rand(size=(input_ids.shape[0],)) < self.classifier_free_ratio
        input_ids[classifier_free_mask] = self.unconditional_input_ids.to(input_ids.device)

        # encode image into latents
        latents = self.encode_image(pixel_values)

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep (forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # get text condition
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # run unet
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        if self.use_pixel_loss:
            latent_pred = self.get_image_pred(model_pred, timesteps, noisy_latents)
            image_pred = self.decode_latents(latent_pred)
            image_pred = torch.clamp(image_pred, -1, 1)
            loss = F.mse_loss(image_pred.float(), pixel_values.float(), reduction="mean")
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()
        self.text_encoder.gradient_checkpointing_enable()


def log_to_wandb(images, gt_images, captions, scores, gt_scores, global_step):
    logger.info("Uploading to wandb")
    images = [wandb.Image(img) for img in images]
    gt_images = [wandb.Image(img) for img in gt_images]
    predictions = [captions, images, gt_images, scores, gt_scores,
                   [global_step] * len(captions)]
    columns = ["prompt", "image", "gt_image", "score", "gt_score", "global_step"]
    data = list(zip(*predictions))
    table = wandb.Table(columns=columns, data=data)
    wandb.log({"test_predictions": table}, commit=False, step=global_step)


def clean_ckpts(output_dir, max_ckpts_to_keep):
    all_ckpts = list(glob(os.path.join(output_dir, f"epoch-*")))
    all_ckpts.sort(key=os.path.getctime)
    if len(all_ckpts) > max_ckpts_to_keep:
        for ckpt in all_ckpts[:-max_ckpts_to_keep]:
            shutil.rmtree(ckpt)


def get_latest_ckpt_path(output_dir):
    all_ckpts = list(glob(os.path.join(output_dir, f"epoch-*")))
    all_ckpts.sort(key=os.path.getctime)
    if len(all_ckpts) > 0:
        return all_ckpts[-1]
    return None


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def extract_from_ckpt_path(ckpt_path):
    ckpt_path = os.path.basename(ckpt_path)
    ints = ckpt_path.replace("epoch-", "").replace("global_step-", "").replace("step-", "").split("_")
    start_epoch, start_step, global_step = [int(i) for i in ints]
    return start_epoch, start_step, global_step


@hydra.main(config_path="../../configs/finetune", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    def clean_output_dir():
        if os.path.exists(cfg.general.output_dir):
            if cfg.general.overwrite_output_dir:
                logger.info(f"Overwriting {cfg.general.output_dir}")
                shutil.rmtree(cfg.general.output_dir)

        logger.info(f"Will write to {os.path.realpath(cfg.general.output_dir)}")
        os.makedirs(cfg.general.output_dir, exist_ok=True)

    def set_logging_level():
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

    logging_dir = os.path.join(cfg.general.output_dir, cfg.general.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=cfg.general.report_to,
        logging_dir=logging_dir,
    )

    if accelerator.is_main_process:
        clean_output_dir()
        print_config(cfg)
        if cfg.debug.activate:
            debug(cfg.debug.port)

    set_logging_level()

    if cfg.training.seed is not None:
        set_seed(cfg.training.seed, device_specific=True)

    logger.info("Loading model and scheduler")
    pipeline = StableDiffusionPipeline.from_pretrained(cfg.training.pretrained_model_name_or_path)
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.training.pretrained_model_name_or_path,
        prediction_type=cfg.training.prediction_type,
        subfolder="scheduler"
    )

    pipeline.safety_checker = None
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet

    if not cfg.training.train_unet:
        logger.info("Freezing unet")
        unet.requires_grad_(False)

    if not cfg.training.train_text_encoder:
        logger.info("Freezing text encoder")
        text_encoder.requires_grad_(False)

    if not cfg.training.train_vae:
        logger.info("Freezing vae")
        vae.requires_grad_(False)

    unconditional_input_ids = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length",
                                        truncation=True, return_tensors="pt").input_ids

    model = Model(
        unet,
        text_encoder,
        vae,
        unconditional_input_ids,
        cfg.training.classifier_free_ratio,
        noise_scheduler,
        cfg.training.use_pixel_loss
    )

    if cfg.training.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.enable_gradient_checkpointing()

    logger.info("Loading CLIP (for validation)")
    clip_processor = CLIPProcessor.from_pretrained(cfg.training.eval_clip_id)
    clip_model = CLIPModel.from_pretrained(cfg.training.eval_clip_id, torch_dtype=torch.float16)
    clip_model.requires_grad_(False)
    clip_model = clip_model.to(accelerator.device)

    logger.info("Loading scheduler (for validation)")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(cfg.training.pretrained_model_name_or_path,
                                                            subfolder="scheduler")
    pipeline.scheduler = scheduler

    logger.info("Initializing optimizer")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.optimizer.learning_rate,
        betas=(cfg.optimizer.adam_beta1, cfg.optimizer.adam_beta2),
        weight_decay=cfg.optimizer.adam_weight_decay,
        eps=cfg.optimizer.adam_epsilon,
    )

    logger.info("Loading datasets")
    dataset = load_dataset(
        cfg.dataset.dataset_name,
        cfg.dataset.dataset_config_name,
        cache_dir=cfg.general.cache_dir,
    )

    if cfg.dataset.max_examples is not None:
        logger.info(f"Limiting dataset to {cfg.dataset.max_examples} examples")
        for split in dataset:
            dataset[split] = dataset[split].select(list(range(min(cfg.dataset.max_examples, len(dataset[split])))))

    logger.info("Configuring dataset preprocessing")

    def tokenize_captions(examples):
        captions = examples[cfg.dataset.caption_column_name]
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids
        return input_ids

    train_transforms = transforms.Compose(
        [
            transforms.Resize(cfg.training.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(cfg.training.resolution),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2. - 1.)
        ]
    )

    def preprocess(examples):
        examples["input_ids"] = tokenize_captions(examples)
        examples["pixel_values"] = [train_transforms(image) for image in examples[cfg.dataset.image_column_name]]
        return examples

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess)
        # TODO change to test
        validation_dataset = dataset["validation"].with_transform(preprocess)

    def collate_fn(examples):
        input_ids = torch.stack([example["input_ids"] for example in examples])

        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        return {"input_ids": input_ids, "pixel_values": pixel_values}

    logger.info("Initializing data loaders")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg.training.train_batch_size,
        num_workers=cfg.training.num_workers
    )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=cfg.training.validation_batch_size,
        num_workers=cfg.training.num_workers
    )

    logger.info("Loading learning rate scheduler")
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    if cfg.training.max_train_steps is None:
        cfg.training.max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.lr_warmup_steps * cfg.training.gradient_accumulation_steps,
        num_training_steps=cfg.training.max_train_steps * cfg.training.gradient_accumulation_steps,
    )

    logger.info("Preparing training with accelerator")
    model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.training.max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.training.num_train_epochs = math.ceil(cfg.training.max_train_steps / num_update_steps_per_epoch)

    def train_step(pixel_values, input_ids):
        loss = model(pixel_values, input_ids)

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss).mean().item()

        # Backpropagate
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        return avg_loss

    @torch.no_grad()
    def get_image_scores(caption, img, gt_img):
        inputs = clip_processor(text=[caption], truncation=True, padding="max_length", max_length=77,
                                images=[img, gt_img],
                                return_tensors="pt").to(accelerator.device)
        image_features = clip_model.get_image_features(inputs["pixel_values"].to(torch.float16))
        text_features = clip_model.get_text_features(inputs["input_ids"])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        prompt_scores = text_features @ image_features.T
        img_score, gt_img_score = prompt_scores[0, 0].item(), prompt_scores[0, 1].item()
        img_score = round(img_score, 3)
        gt_img_score = round(gt_img_score, 3)
        return img_score, gt_img_score

    def generate_validation():
        valid_progress_bar = tqdm(disable=not accelerator.is_main_process)
        valid_progress_bar.set_description("#Batches Generated")

        captions, images, gt_images = [], [], []
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(cfg.training.seed)

        for eval_step, val_batch in enumerate(validation_dataloader):
            batch_captions = tokenizer.batch_decode(val_batch["input_ids"], skip_special_tokens=True)
            batch_images = pipeline(batch_captions,
                                    guidance_scale=cfg.training.gs,
                                    generator=generator,
                                    num_inference_steps=cfg.training.n_eval_steps).images
            captions.extend(batch_captions)
            images.extend(batch_images)
            gt_images.extend(pixel_values_to_pil_images(val_batch["pixel_values"]))
            valid_progress_bar.update(1)
            if len(captions) >= cfg.training.max_batches_to_generate * cfg.training.validation_batch_size:
                break

        return captions, images, gt_images

    def run_clip_score(captions, images, gt_images):
        scores, gt_scores = [], []
        for caption, img, gt_img in zip(captions, images, gt_images):
            img_score, gt_img_score = get_image_scores(caption, img, gt_img)
            scores.append(img_score)
            gt_scores.append(gt_img_score)
        return scores, gt_scores

    def gather_iterable(it):
        output_objects = [None for _ in range(accelerator.num_processes)]
        torch.distributed.all_gather_object(output_objects, it)
        return flatten(output_objects)

    def evaluate():
        logger.info("***** Generating from Validation *****")
        model.eval()
        pipeline.set_progress_bar_config(disable=True)

        captions, images, gt_images = generate_validation()
        scores, gt_scores = run_clip_score(captions, images, gt_images)

        log_data = [captions, images, gt_images, scores, gt_scores]
        captions, images, gt_images, scores, gt_scores = [gather_iterable(dp) for dp in log_data]

        # Log predictions
        if cfg.general.report_to == "wandb" and accelerator.is_main_process:
            log_to_wandb(images, gt_images, captions, scores, gt_scores, global_step)

        logger.info(f"Finished Validation")
        torch.cuda.empty_cache()
        model.train()
        accelerator.wait_for_everyone()

    start_epoch, start_step, continuing, global_step = None, None, False, 0
    if cfg.training.continue_from_checkpoint:
        ckpt_path = get_latest_ckpt_path(cfg.general.output_dir)
        if ckpt_path is not None:
            continuing = True
            start_epoch, start_step, global_step = extract_from_ckpt_path(ckpt_path)
            logger.info(f"Loading state from {ckpt_path} - epoch {start_epoch}, step {start_step}, global_step {global_step}")
            accelerator.load_state(ckpt_path)

    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=OmegaConf.to_object(cfg))

    logger.info("Training")
    total_batch_size = cfg.training.train_batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  GPU memory usage before training {get_allocated_cuda_memory()}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.training.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.training.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.training.max_train_steps}")
    logger.info(f"  Mixed precision = {cfg.training.mixed_precision}")
    logger.info(f"  Classifier free ratio = {cfg.training.classifier_free_ratio}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.training.max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(cfg.training.num_train_epochs):
        model.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if continuing and epoch < start_epoch or (epoch == start_epoch and step < start_step):
                progress_bar.update(1)
                progress_bar.set_postfix(**{"status": "skipping"})
                continue
            with accelerator.accumulate(model):

                is_start_train = step == 0 and epoch == 0
                should_eval = accelerator.sync_gradients and global_step % cfg.training.validate_steps == 0
                if is_start_train or should_eval:
                    evaluate()

                avg_loss = train_step(**batch)
                torch.cuda.empty_cache()

                # Checks if the accelerator has performed an optimization step behind the scenes
                train_loss += avg_loss / cfg.training.gradient_accumulation_steps
                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)
                    accelerator.log(
                        {
                            "train_loss": train_loss,
                            "train_epoch": epoch,
                            "train_step": global_step,
                            "lr": lr_scheduler.get_last_lr()[0]
                        },
                        step=global_step)
                    train_loss = 0.0

                cuda_gb_allocated = get_allocated_cuda_memory()
                logs = {
                    "step_loss": avg_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "mem": cuda_gb_allocated,
                    "epoch": epoch
                }
                progress_bar.set_postfix(**logs)

                # Save checkpoint
                if accelerator.sync_gradients and global_step % cfg.training.save_steps == 0:
                    save_dir = cfg.general.output_dir
                    os.makedirs(save_dir, exist_ok=True)
                    save_dir = os.path.join(cfg.general.output_dir,
                                            f"epoch-{epoch}_step-{step}_global_step-{global_step}")
                    accelerator.save_state(save_dir)
                    if accelerator.is_main_process:
                        clean_ckpts(cfg.general.output_dir, cfg.general.max_ckpts_to_keep)

            if global_step >= cfg.training.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()
