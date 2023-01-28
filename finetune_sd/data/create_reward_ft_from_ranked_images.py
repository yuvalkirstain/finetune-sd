import os.path
import random
from multiprocessing import Pool
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from img2dataset import download
from omegaconf import DictConfig
from utils import parquet2dataset, dataset2images, filter_dataset_by_text
import hydra


@hydra.main(config_path="../../configs/data", config_name="reward_config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Starting...")

    print("Loading datasets...")
    eval_dataset = load_dataset(cfg.paths.eval_dataset_name)["train"]
    ranking_dataset = load_dataset(cfg.paths.ranking_dataset_name)["train"]

    print("Filtering datasets...")
    eval_prompts = set(eval_dataset[cfg.dataset_columns.text_col])
    decision_ranking_dataset = ranking_dataset.filter(
        lambda x:
        x[cfg.dataset_columns.best_uid_col] != 'none' and
        x[cfg.dataset_columns.text_col] not in eval_prompts
    )
    decision_ranking_dataset = filter_dataset_by_text(decision_ranking_dataset, cfg.dataset_columns.text_col)
    df = decision_ranking_dataset.to_pandas()
    df = df.sort_values("created_at").drop_duplicates("prompt")

    print("Keeping only one pair per prompt...")
    df[cfg.dataset_columns.uid_bad_col] = df.apply(lambda x: random.choice(
        list(set({x["image_1_uid"], x["image_2_uid"], x["image_3_uid"], x["image_4_uid"]}) - set(
            {x[cfg.dataset_columns.best_uid_col]}))), axis=1)

    print("Adding URLs...")
    df = df[[cfg.dataset_columns.text_col, cfg.dataset_columns.uid_bad_col, cfg.dataset_columns.best_uid_col]]
    df[cfg.dataset_columns.url_bad_col] = df.apply(
        lambda
            x: f"https://text-to-image-human-preferences.s3.us-east-2.amazonaws.com/images/{x[cfg.dataset_columns.uid_bad_col]}.png",
        axis=1)
    df[cfg.dataset_columns.url_good_col] = df.apply(
        lambda
            x: f"https://text-to-image-human-preferences.s3.us-east-2.amazonaws.com/images/{x[cfg.dataset_columns.best_uid_col]}.png",
        axis=1)

    print("Saving urls for img2dataset...")
    df.to_parquet(cfg.paths.urls_out_path)

    if not os.path.exists(cfg.paths.images_good_out_path):
        print("Downloading images...")
        download(
            processes_count=cfg.download.process_count,
            thread_count=cfg.download.thread_count,
            url_list=cfg.paths.urls_out_path,
            image_size=cfg.download.image_size,
            output_folder=cfg.paths.images_good_out_path,
            resize_mode=cfg.download.resize_mode,
            resize_only_if_bigger=True,
            output_format="parquet",
            input_format="parquet",
            url_col=cfg.dataset_columns.url_good_col,
            caption_col=cfg.dataset_columns.text_col,
            number_sample_per_shard=cfg.download.num_samples_per_shard,
            distributor="multiprocessing",
            min_image_size=cfg.download.image_size
        )
    else:
        print("Good images already downloaded.")

    if not os.path.exists(cfg.paths.images_bad_out_path):
        print("Downloading images...")
        download(
            processes_count=cfg.download.process_count,
            thread_count=cfg.download.thread_count,
            url_list=cfg.paths.urls_out_path,
            image_size=cfg.download.image_size,
            output_folder=cfg.paths.images_bad_out_path,
            resize_mode=cfg.download.resize_mode,
            resize_only_if_bigger=True,
            output_format="parquet",
            input_format="parquet",
            url_col=cfg.dataset_columns.url_bad_col,
            caption_col=cfg.dataset_columns.text_col,
            number_sample_per_shard=cfg.download.num_samples_per_shard,
            distributor="multiprocessing",
            min_image_size=cfg.download.image_size
        )
    else:
        print("Bad images already downloaded.")

    if not os.path.exists(cfg.paths.dataset_out_path):
        print("Reading downloaded images...")
        dataset = parquet2dataset(cfg.paths.images_good_out_path)
        dataset_bad = parquet2dataset(cfg.paths.images_bad_out_path)

        print("Filtering errors...")
        dataset = dataset.filter(lambda x: x['error_message'] is None)
        dataset_bad = dataset_bad.filter(lambda x: x['error_message'] is None)
        prompts_bad = set(dataset_bad['caption'])
        dataset = dataset.filter(lambda x: x['caption'] in prompts_bad)

        with Pool(processes=cfg.download.process_count) as pool:
            print("getting good images")
            images_good = dataset2images(dataset, pool, "jpg")
            print("getting bad images")
            images_bad = dataset2images(dataset_bad, pool, "jpg")

        prompt2bad_image = {dataset_bad[i]["caption"]: images_bad[i] for i in list(range(len(dataset_bad)))}
        prompt2bad_url = {x["caption"]: x["url"] for x in dataset_bad}

        print("creating dataset with images")
        dataset = Dataset.from_dict(
            {
                "good_image": images_good,
                "bad_image": [prompt2bad_image[x["caption"]] for x in dataset],
                "text": dataset["caption"],
                "good_url": dataset["url"],
                "bad_url": [prompt2bad_url[x["caption"]] for x in dataset],
            }
        )
        print(f"splitting dataset to train-test - {cfg.others.ratio=}")
        dataset = dataset.train_test_split(cfg.others.ratio)

        print(f"saving to disk {cfg.paths.dataset_out_path}")
        dataset.save_to_disk(cfg.paths.dataset_out_path)
    else:
        print("Dataset already exists.. Reading from disk")
        dataset = load_from_disk(cfg.paths.dataset_out_path)

    print(f"pushing to hub yuvalkirstain/{cfg.paths.dataset_hub_name}")
    dataset.push_to_hub(cfg.paths.dataset_hub_name)


if __name__ == '__main__':
    main()
