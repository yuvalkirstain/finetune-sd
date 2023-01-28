from glob import glob
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk


def parquet2dataset(parquet_path: str):
    datasets = []
    for path in sorted(glob(f"{parquet_path}/*.parquet")):
        datasets.append(load_dataset("parquet", data_files=path)["train"])
    dataset = concatenate_datasets(datasets)
    return dataset


def bytes2image(bytes: bytes):
    image = Image.open(BytesIO(bytes))
    image = image.convert("RGB")
    return image


def dataset2images(dataset, pool, col):
    image_bytes = dataset[col]
    images = list(tqdm(pool.imap(bytes2image, image_bytes), total=len(image_bytes)))
    return images


def filter_dataset_by_text(datset, text_col):
    dataset = datset.filter(
        lambda x:
        x[text_col].strip().lower() != "A mature poodle toy.".lower() and
        x[text_col].strip().lower() != "A mature toy poodle.".lower() and
        "busty" not in x[text_col].lower() and
        "boobs" not in x[text_col].lower() and
        "gay" not in x[text_col].lower() and
        "shirtless" not in x[text_col].lower() and
        "nude" not in x[text_col].lower() and
        "revealing" not in x[text_col].lower() and
        "naked" not in x[text_col].lower() and
        "breasts" not in x[text_col].lower() and
        "amouranth" not in x[text_col].lower() and
        "nigger" not in x[text_col].lower() and
        "sussy" not in x[text_col].lower() and
        "tits" not in x[text_col].lower() and
        "lingerie" not in x[text_col].lower() and
        "trump" not in x[text_col].lower() and
        "sex" not in x[text_col].lower() and
        "bikini" not in x[text_col].lower() and
        "netanyahu" not in x[text_col].lower() and
        "jewish" not in x[text_col].lower() and
        "putin" not in x[text_col].lower() and
        len(x[text_col].strip()) > 2
    )
    return dataset
