import os.path
from multiprocessing import Pool
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from img2dataset import download
from utils import parquet2dataset, dataset2images, filter_dataset_by_text


def main():
    print("Starting...")
    urls_out_path = f"data/urls/PickaPic.parquet"
    images_out_path = f"data/images/PickaPic.parquet"
    dataset_out_path = f"data/datasets/PickaPic.ds"
    dataset_hub_name = "PickaPic-ft-ranked"

    process_count = 16
    thread_count = 32
    image_size = 512
    resize_mode = "center_crop"
    num_samples_per_shard = 1000
    text_col = "prompt"
    url_col = "url"

    ranking_dataset_name = "yuvalkirstain/PickaPic-rankings"
    images_dataset_name = "yuvalkirstain/PickaPic-images"
    eval_dataset_name = "yuvalkirstain/PickaPic-random-prompts"

    print("Loading datasets...")
    eval_dataset = load_dataset(eval_dataset_name)["train"]
    ranking_dataset = load_dataset(ranking_dataset_name)["train"]
    images_dataset = load_dataset(images_dataset_name)["train"]

    print("Filtering datasets...")
    eval_prompts = set(eval_dataset["prompt"])
    decision_ranking_dataset = ranking_dataset.filter(
        lambda x:
        x['best_image_uid'] != 'none'
    )

    best_image_uids = set(decision_ranking_dataset['best_image_uid'])
    best_images_dataset = images_dataset.filter(
        lambda x:
        x['image_uid'] in best_image_uids and
        x['prompt'] not in eval_prompts
    )
    best_images_dataset = filter_dataset_by_text(best_images_dataset, text_col)

    print("Saving urls for img2dataset...")
    best_images_dataset.to_parquet(urls_out_path)

    if not os.path.exists(images_out_path):
        print("Downloading images...")
        download(
            processes_count=process_count,
            thread_count=thread_count,
            url_list=urls_out_path,
            image_size=image_size,
            output_folder=images_out_path,
            resize_mode=resize_mode,
            resize_only_if_bigger=True,
            output_format="parquet",
            input_format="parquet",
            url_col=url_col,
            caption_col=text_col,
            number_sample_per_shard=num_samples_per_shard,
            distributor="multiprocessing",
            min_image_size=image_size
        )
    else:
        print("Images already downloaded.")

    if not os.path.exists(dataset_out_path):
        print("Reading downloaded images...")
        dataset = parquet2dataset(images_out_path)
        print("Filtering errors...")
        dataset = dataset.filter(lambda x: x['error_message'] is None)

        with Pool(processes=process_count) as pool:
            print("getting images")
            images = dataset2images(dataset, pool)

        print("creating dataset with images")
        dataset = Dataset.from_dict(
            {
                "train": {
                    "image": images,
                    "text": dataset["caption"],
                    "width": dataset["width"],
                    "height": dataset["height"],
                    "url": dataset["url"],
                },
                "validation": {
                    "image": [None] * len(eval_dataset),
                    "text": eval_dataset["prompt"],
                    "width": [None] * len(eval_dataset),
                    "height": [None] * len(eval_dataset),
                    "url": [None] * len(eval_dataset),
                }
            }
        )

        print(f"saving to disk {dataset_out_path}")
        dataset.save_to_disk(dataset_out_path)
    else:
        print("Dataset already exists.. Reading from disk")
        dataset = load_from_disk(dataset_out_path)

    print(f"pushing to hub yuvalkirstain/{dataset_hub_name}")
    dataset.push_to_hub(dataset_hub_name)


if __name__ == '__main__':
    main()
