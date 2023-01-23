import json
import math
import os
import hydra
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler
from datasets import load_dataset
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    get_scheduler,
    DPMSolverMultistepScheduler,
)
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from utils import (
    print_config,
    debug,
    pixel_values_to_pil_images,
    infer_weight_dtype,
    log_to_wandb,
    load_clip_for_eval,
    set_logging_level,
    clean_output_dir,
    get_allocated_cuda_memory,
    clean_ckpts,
    gather_iterable,
    get_latest_ckpt_path,
    extract_from_ckpt_path,
    set_seed, set_mixed_precision_in_hf_config,
)
from model import Model
from dataset import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = get_logger(__name__)


@hydra.main(config_path="../../configs/finetune", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.training.add_face_loss:
        assert cfg.training.use_pixel_loss, "Face loss requires pixel loss"

    logging_dir = os.path.join(cfg.general.output_dir, cfg.general.logging_dir)
    hf_ds_config = json.load(open(cfg.training.deepspeed_config_path))
    set_mixed_precision_in_hf_config(hf_ds_config, cfg.training.mixed_precision)
    deepspeed_plugin = DeepSpeedPlugin(
        hf_ds_config=hf_ds_config,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
    )

    cfg.training.deepspeed = OmegaConf.create(deepspeed_plugin.deepspeed_config)

    accelerator = Accelerator(
        deepspeed_plugin=deepspeed_plugin,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=cfg.general.report_to,
        logging_dir=logging_dir,
    )

    set_logging_level(accelerator.is_local_main_process)

    if accelerator.is_main_process:
        clean_output_dir(cfg.general.output_dir, cfg.general.overwrite_output_dir)

    if accelerator.is_main_process:
        if cfg.debug.activate:
            debug(cfg.debug.port)

    base_seed = cfg.training.seed if cfg.training.seed is not None else 42
    set_seed(base_seed)

    logger.info("Loading model and training noise scheduler")
    pipeline = StableDiffusionPipeline.from_pretrained(cfg.training.pretrained_model_name_or_path)
    noise_scheduler = DDPMScheduler.from_pretrained(
        cfg.training.pretrained_model_name_or_path,
        prediction_type=cfg.training.prediction_type,
        subfolder="scheduler"
    )
    old_safety_checker = pipeline.safety_checker
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

    unconditional_input_ids = tokenizer(
        "",
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    weight_dtype = infer_weight_dtype(accelerator.mixed_precision)

    model = Model(
        unet,
        text_encoder,
        vae,
        unconditional_input_ids,
        cfg.training.classifier_free_ratio,
        noise_scheduler,
        cfg.training.use_pixel_loss,
        weight_dtype
    )

    if cfg.training.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.enable_gradient_checkpointing()

    logger.info("Loading CLIP (for validation)")
    clip_model, clip_processor = load_clip_for_eval(cfg.training.eval_clip_id, accelerator.device)

    logger.info("Loading scheduler (for validation + save)")
    scheduler = DPMSolverMultistepScheduler.from_pretrained(cfg.training.pretrained_model_name_or_path,
                                                            subfolder="scheduler")
    pipeline.scheduler = scheduler

    logger.info("Loading HF datasets")
    dataset = load_dataset(
        cfg.dataset.dataset_name,
        cfg.dataset.dataset_config_name,
        cache_dir=cfg.general.cache_dir,
    )

    if cfg.dataset.max_examples is not None:
        logger.info(f"Limiting dataset to {cfg.dataset.max_examples} examples")
        for split in dataset:
            dataset[split] = dataset[split].select(list(range(min(cfg.dataset.max_examples, len(dataset[split])))))

    logger.info("Creating training datasets")
    train_dataset = Dataset(
        dataset["train"],
        tokenizer,
        cfg.training.resolution,
        cfg.training.add_face_loss,
        cfg.dataset.caption_column_name,
        cfg.dataset.image_column_name
    )
    # TODO change to test
    validation_dataset = Dataset(
        dataset["train"],
        tokenizer,
        cfg.training.resolution,
        False,
        cfg.dataset.caption_column_name,
        cfg.dataset.image_column_name
    )

    logger.info("Initializing data loaders")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        batch_size=cfg.training.train_batch_size,
        num_workers=cfg.training.num_workers
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        collate_fn=validation_dataset.collate_fn,
        batch_size=cfg.training.validation_batch_size,
        num_workers=cfg.training.num_workers
    )

    logger.info("Initializing optimizer")
    optimizer = DummyOptim([p for p in model.parameters() if p.requires_grad], lr=cfg.optimizer.learning_rate)

    logger.info("Loading learning rate scheduler")
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    if cfg.training.max_train_steps is None:
        cfg.training.max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    total_warmup_steps = cfg.lr_scheduler.lr_warmup_steps  # * cfg.training.gradient_accumulation_steps
    total_training_steps = cfg.training.max_train_steps  # * cfg.training.gradient_accumulation_steps
    lr_scheduler = DummyScheduler(
        optimizer,
        total_num_steps=total_training_steps * accelerator.num_processes,
        warmup_num_steps=total_warmup_steps,
        warmup_max_lr=cfg.optimizer.learning_rate,
    )

    logger.info("Preparing with accelerator")
    model, optimizer, train_dataloader, validation_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.training.max_train_steps = cfg.training.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.training.num_train_epochs = math.ceil(cfg.training.max_train_steps / num_update_steps_per_epoch)

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

    def evaluate():
        logger.info("***** Generating from Validation *****")
        model.eval()
        pipeline.set_progress_bar_config(disable=True)

        captions, images, gt_images = generate_validation()
        scores, gt_scores = run_clip_score(captions, images, gt_images)

        log_data = [captions, images, gt_images, scores, gt_scores]
        captions, images, gt_images, scores, gt_scores = [gather_iterable(dp, accelerator.num_processes) for dp in
                                                          log_data]

        # Log predictions
        if cfg.general.report_to == "wandb" and accelerator.is_main_process:
            log_to_wandb(images, gt_images, captions, scores, gt_scores, global_step)

        logger.info(f"Finished Validation")
        model.train()
        accelerator.wait_for_everyone()

    start_epoch, start_step, continuing, global_step = None, None, False, 0
    if cfg.training.continue_from_checkpoint:
        ckpt_path = get_latest_ckpt_path(cfg.general.output_dir)
        if ckpt_path is not None:
            continuing = True
            start_epoch, start_step, global_step = extract_from_ckpt_path(ckpt_path)
            logger.info(f"Loading {ckpt_path}: epoch={start_epoch} step={start_step} global_step={global_step}")
            accelerator.load_state(ckpt_path)

    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=OmegaConf.to_object(cfg))
        print_config(cfg)

    if cfg.debug.activate:
        accelerator.wait_for_everyone()

    logger.info("Training")
    total_batch_size = cfg.training.train_batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  GPU memory usage before training {get_allocated_cuda_memory(accelerator.device)}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.training.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.training.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"  Total warmup steps = {total_warmup_steps}")
    logger.info(f"  Total training steps = {total_training_steps}")
    logger.info(f"  Total optimization steps = {cfg.training.max_train_steps}")
    logger.info(f"  Mixed precision = {accelerator.mixed_precision}")
    logger.info(f"  Classifier free ratio = {cfg.training.classifier_free_ratio}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.training.max_train_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(cfg.training.num_train_epochs):
        model.train()
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):

            if cfg.debug.activate:
                accelerator.wait_for_everyone()

            if continuing and epoch < start_epoch or (epoch == start_epoch and step < start_step):
                progress_bar.update(1)
                progress_bar.set_postfix(**{"status": "skipping"})
                continue

            eval_on_start_train = step == 0 and epoch == 0 and cfg.training.eval_on_start_train
            should_eval = step % cfg.training.gradient_accumulation_steps == 0 and global_step % cfg.training.validate_steps == 0
            if eval_on_start_train or should_eval:
                evaluate()

            with accelerator.accumulate(model):
                loss = model(**batch)
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss).mean().item()
                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
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

            cuda_gb_allocated = get_allocated_cuda_memory(accelerator.device)
            logs = {
                "step_loss": avg_loss,
                "mem": cuda_gb_allocated,
                "step": step,
                "epoch": epoch,
                "global_step": global_step,
            }
            if global_step > 0:
                logs["lr"] = lr_scheduler.get_last_lr()[0]

            progress_bar.set_postfix(**logs)

            # Save checkpoint
            if accelerator.sync_gradients and global_step % cfg.training.save_steps == 0:
                save_dir = cfg.general.output_dir
                os.makedirs(save_dir, exist_ok=True)
                save_dir = os.path.join(cfg.general.output_dir,
                                        f"epoch-{epoch}_step-{step}_global_step-{global_step}")
                accelerator.save_state(save_dir)
                logger.info(f"Saved checkpoint to {save_dir}")
                if accelerator.is_main_process:
                    clean_ckpts(cfg.general.output_dir, cfg.general.max_ckpts_to_keep)

            if global_step >= cfg.training.max_train_steps:
                break

        if global_step >= cfg.training.max_train_steps:
            break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)

        pipeline.safety_checker = old_safety_checker
        pipeline.unet = model.unet
        pipeline.text_encoder = model.text_encoder
        pipeline.vae = model.vae

        save_dir = os.path.join(cfg.general.output_dir, f"epoch-{epoch}_step-{step}_global_step-{global_step}-final")
        clean_ckpts(cfg.general.output_dir, cfg.general.max_ckpts_to_keep)
        pipeline.save_pretrained(save_dir)
        logger.info(f"Saved pipeline to {save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
