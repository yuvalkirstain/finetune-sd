from omegaconf import DictConfig, OmegaConf
import rich.tree
import rich.syntax
from accelerate.logging import get_logger
from accelerate.utils import set_seed as accelerate_set_seed
from PIL import Image
import torch
import wandb
from glob import glob
import os
import shutil
import datasets
import diffusers
import transformers
from transformers import CLIPModel, CLIPProcessor

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


def get_allocated_cuda_memory(device):
    return round(torch.cuda.max_memory_allocated(device) / 1024 / 1024 / 1024, 2)


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
        if "final" in all_ckpts[-1]:
            all_ckpts.pop()
        return all_ckpts[-1]
    return None


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def extract_from_ckpt_path(ckpt_path):
    ckpt_path = os.path.basename(ckpt_path)
    ints = ckpt_path.replace("epoch-", "").replace("global_step-", "").replace("step-", "").split("_")
    start_epoch, start_step, global_step = [int(i) for i in ints]
    return start_epoch, start_step, global_step


def convert_to_RGB(example):
    return example.convert("RGB")


def clean_output_dir(output_dir, overwrite_output_dir):
    if os.path.exists(output_dir):
        if overwrite_output_dir:
            logger.info(f"Overwriting {output_dir}")
            shutil.rmtree(output_dir)

    logger.info(f"Will write to {os.path.realpath(output_dir)}")
    os.makedirs(output_dir, exist_ok=True)


def set_logging_level(is_local_main_process):
    if is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def infer_weight_dtype(mixed_precision):
    if mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif mixed_precision == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    return weight_dtype


def load_clip_for_eval(eval_clip_id, device):
    clip_processor = CLIPProcessor.from_pretrained(eval_clip_id)
    clip_model = CLIPModel.from_pretrained(eval_clip_id, torch_dtype=torch.float16)
    clip_model.requires_grad_(False)
    clip_model = clip_model.to(device)
    return clip_model, clip_processor


def gather_iterable(it, num_processes):
    output_objects = [None for _ in range(num_processes)]
    torch.distributed.all_gather_object(output_objects, it)
    return flatten(output_objects)


def set_seed(seed):
    base_seed = seed if seed is not None else 42
    logger.info(f"Setting seed {base_seed}")
    accelerate_set_seed(base_seed, device_specific=True)


def set_mixed_precision_in_hf_config(hf_ds_config, mixed_precision):
    if mixed_precision == "bf16":
        hf_ds_config["fp16"]["enabled"] = False
        hf_ds_config["bf16"]["enabled"] = True
    elif mixed_precision == "fp16":
        hf_ds_config["fp16"]["enabled"] = True
        hf_ds_config["bf16"]["enabled"] = False
    else:
        hf_ds_config["fp16"]["enabled"] = False
        hf_ds_config["bf16"]["enabled"] = False
