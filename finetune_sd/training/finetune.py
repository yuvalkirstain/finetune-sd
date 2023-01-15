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


class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.collected_params = None

        self.decay = decay
        self.optimization_step = 0

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1

        # Compute the decay factor for the exponential moving average.
        value = (1 + self.optimization_step) / (10 + self.optimization_step)
        one_minus_decay = 1 - min(self.decay, value)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                s_param.sub_(one_minus_decay * (s_param - param))
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:
        r"""
        Returns the state of the ExponentialMovingAverage as a dict.
        This method is used by accelerate during checkpointing to save the ema state dict.
        """
        # Following PyTorch conventions, references to tensors are returned:
        # "returns a reference to the state and not its copy!" -
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
        return {
            "decay": self.decay,
            "optimization_step": self.optimization_step,
            "shadow_params": self.shadow_params,
            "collected_params": self.collected_params,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        r"""
        Loads the ExponentialMovingAverage state.
        This method is used by accelerate during checkpointing to save the ema state dict.
        Args:
            state_dict (dict): EMA state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)

        self.decay = state_dict["decay"]
        if self.decay < 0.0 or self.decay > 1.0:
            raise ValueError("Decay must be between 0 and 1")

        self.optimization_step = state_dict["optimization_step"]
        if not isinstance(self.optimization_step, int):
            raise ValueError("Invalid optimization_step")

        self.shadow_params = state_dict["shadow_params"]
        if not isinstance(self.shadow_params, list):
            raise ValueError("shadow_params must be a list")
        if not all(isinstance(p, torch.Tensor) for p in self.shadow_params):
            raise ValueError("shadow_params must all be Tensors")

        self.collected_params = state_dict["collected_params"]
        if self.collected_params is not None:
            if not isinstance(self.collected_params, list):
                raise ValueError("collected_params must be a list")
            if not all(isinstance(p, torch.Tensor) for p in self.collected_params):
                raise ValueError("collected_params must all be Tensors")
            if len(self.collected_params) != len(self.shadow_params):
                raise ValueError("collected_params and shadow_params must have the same length")


def get_allocated_cuda_memory():
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, 2)


@hydra.main(config_path="../../configs/finetune", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging_dir = os.path.join(cfg.general.output_dir, cfg.general.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=cfg.general.report_to,
        logging_dir=logging_dir,
    )

    if accelerator.is_main_process:
        if os.path.exists(cfg.general.output_dir):
            if cfg.general.overwrite_output_dir:
                shutil.rmtree(cfg.general.output_dir)
            else:
                raise ValueError(f"Output directory {cfg.general.output_dir} already exists!")

        logger.info(f"Will write to {os.path.realpath(cfg.general.output_dir)}")
        os.makedirs(cfg.general.output_dir, exist_ok=True)

        print_config(cfg)

        if cfg.debug.activate:
            debug(cfg.debug.port)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if cfg.training.seed is not None:
        set_seed(cfg.training.seed, device_specific=True)

    logger.info("Loading model")
    pipeline = StableDiffusionPipeline.from_pretrained(cfg.training.pretrained_model_name_or_path)
    pipeline.safety_checker = None
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    vae = pipeline.vae
    unet = pipeline.unet

    if cfg.training.use_ema:
        ema_unet = EMAModel(unet)

    logger.info("Loaded model")

    if cfg.training.enable_xformers_memory_efficient_attention:
        logger.info("Enabling memory efficient attention")
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if cfg.training.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        unet.enable_gradient_checkpointing()

    logger.info("Freezing text encoder and vae")
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    logger.info("Loading CLIP (for validation)")
    clip_processor = CLIPProcessor.from_pretrained(cfg.training.eval_clip_id)
    clip_model = CLIPModel.from_pretrained(cfg.training.eval_clip_id, torch_dtype=torch.float16)
    clip_model.requires_grad_(False)
    clip_model = clip_model.to(accelerator.device)

    logger.info("Loading scheduler (for validation)")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(cfg.training.pretrained_model_name_or_path,
                                                            subfolder="scheduler")
    pipeline.scheduler = scheduler

    logger.info("Loading scheduler (for training")
    noise_scheduler = DDPMScheduler.from_pretrained(cfg.training.pretrained_model_name_or_path, subfolder="scheduler")

    logger.info("Initializing optimizer")
    optimizer = torch.optim.AdamW(
        unet.parameters(),
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

    logger.info("Configuring dataset preprocessing")
    image_column_name = cfg.dataset.image_column_name
    caption_column_name = cfg.dataset.caption_column_name

    def tokenize_captions(examples):
        captions = examples[caption_column_name]
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
        examples["pixel_values"] = [train_transforms(image) for image in examples[image_column_name]]
        return examples

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess)
        # TODO change to test
        validation_dataset = dataset["train"].with_transform(preprocess)

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
    unet, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, validation_dataloader, lr_scheduler
    )

    if cfg.training.use_ema:
        accelerator.register_for_checkpointing(ema_unet)

    weight_dtype = torch.float32
    if cfg.training.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif cfg.training.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info(f"Moving text encoder and vae to GPU with weight dtype {weight_dtype}")
    text_encoder = text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae = vae.to(accelerator.device, dtype=weight_dtype)

    pipeline.unet = unet

    logger.info(f"GPU memory usage before training {get_allocated_cuda_memory()}")

    if cfg.training.use_ema:
        ema_unet.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.training.max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.training.num_train_epochs = math.ceil(cfg.training.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=OmegaConf.to_object(cfg))

    logger.info("Preparing unconditional input")

    def train_forward(batch):
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]

        # classifier free
        unconditional_input_ids = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length",
                                            truncation=True, return_tensors="pt").input_ids
        unconditional_input_ids = unconditional_input_ids.to(accelerator.device)
        classifier_free_mask = torch.rand(size=(input_ids.shape[0],)) < cfg.training.classifier_free_ratio
        input_ids[classifier_free_mask] = unconditional_input_ids

        latents = vae.encode(pixel_values.to(weight_dtype)).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(input_ids)[0]

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # Predict the noise residual and compute loss
        model_out = unet(noisy_latents, timesteps, encoder_hidden_states)
        model_pred = model_out.sample
        return model_pred, target

    def train_step(batch):
        model_pred, target = train_forward(batch)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        # Gather the losses across all processes for logging (if we use distributed training).
        avg_loss = accelerator.gather(loss).mean().item()

        # Backpropagate
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(unet.parameters(), cfg.optimizer.max_grad_norm)
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

    def evaluate():
        logger.info("***** Generating from Validation *****")
        unet.eval()
        pipeline.set_progress_bar_config(disable=True)

        valid_progress_bar = tqdm(disable=not accelerator.is_main_process)
        valid_progress_bar.set_description("#Batches Generated")
        captions, images, gt_images = [], [], []
        generator = torch.Generator(device=accelerator.device)

        for eval_step, val_batch in enumerate(validation_dataloader):
            batch_captions = tokenizer.batch_decode(val_batch["input_ids"], skip_special_tokens=True)
            generator.manual_seed(cfg.training.seed)
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

        scores, gt_scores = [], []
        for caption, img, gt_img in zip(captions, images, gt_images):
            img_score, gt_img_score = get_image_scores(caption, img, gt_img)
            scores.append(img_score)
            gt_scores.append(gt_img_score)

        # Log predictions
        if cfg.general.report_to == "wandb" and accelerator.is_main_process:
            logger.info("Uploading to wandb")
            images = [wandb.Image(img) for img in images]
            gt_images = [wandb.Image(img) for img in gt_images]
            predictions = [captions, images, gt_images, scores, gt_scores,
                           [global_step] * len(captions)]
            columns = ["prompt", "image", "gt_image", "score", "gt_score", "global_step"]
            data = list(zip(*predictions))
            table = wandb.Table(columns=columns, data=data)
            wandb.log({"test_predictions": table}, commit=False, step=global_step)

        logger.info(f"Finished Validation")
        torch.cuda.empty_cache()
        unet.train()
        accelerator.wait_for_everyone()

    def clean_ckpts():
        all_ckpts = list(glob(os.path.join(cfg.general.output_dir, f"global_step*")))
        all_ckpts.sort(key=os.path.getctime)
        if len(all_ckpts) > cfg.general.max_ckpts_to_keep:
            for ckpt in all_ckpts[:-cfg.general.max_ckpts_to_keep]:
                shutil.rmtree(ckpt)

    logger.info("Training")
    # Train!
    total_batch_size = cfg.training.train_batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.training.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.training.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.training.max_train_steps}")
    logger.info(f"  Using weight_dtype = {weight_dtype}")
    logger.info(f"  Using classifier_free_ratio = {cfg.training.classifier_free_ratio}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.training.max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(cfg.training.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                avg_loss = train_step(batch)

                # Checks if the accelerator has performed an optimization step behind the scenes
                train_loss += avg_loss / cfg.training.gradient_accumulation_steps
                if accelerator.sync_gradients:
                    if cfg.training.use_ema:
                        ema_unet.step(unet.parameters())
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
                    if cfg.training.use_ema:
                        # TODO not sure if this is the right way to do it
                        ema_unet.copy_to(unet.parameters())
                    save_dir = cfg.general.output_dir
                    os.makedirs(save_dir, exist_ok=True)
                    save_dir = os.path.join(cfg.general.output_dir, f"global_step_{global_step}")
                    accelerator.save_state(save_dir)
                    if accelerator.is_main_process:
                        clean_ckpts()

                # Evaluate
                is_start_train = step == 0 and epoch == 0
                should_eval = accelerator.sync_gradients and global_step % cfg.training.validate_steps == 0
                if is_start_train or should_eval:
                    evaluate()

            if global_step >= cfg.training.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()
