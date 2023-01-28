import os.path
import random
from multiprocessing import Pool
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from img2dataset import download
from utils import parquet2dataset, dataset2images, filter_dataset_by_text


def main():
    print("Starting...")
    urls_out_path = f"data/urls/PickaPic_reward.parquet"
    images_good_out_path = f"data/images/PickaPic_reward_good.parquet"
    images_bad_out_path = f"data/images/PickaPic_reward_bad.parquet"
    dataset_out_path = f"data/datasets/PickaPic_reward.ds"
    dataset_hub_name = "PickaPic-ft-pairwise"

    process_count = 16
    thread_count = 32
    image_size = 512
    resize_mode = "center_crop"
    num_samples_per_shard = 1000
    text_col = "prompt"
    best_uid_col = "best_image_uid"
    uid_bad_col = "bad_image_uid"
    url_good_col = "url_good"
    url_bad_col = "url_bad"

    ratio = 0.2

    ranking_dataset_name = "yuvalkirstain/PickaPic-rankings"
    eval_dataset_name = "yuvalkirstain/PickaPic-random-prompts"

    print("Loading datasets...")
    eval_dataset = load_dataset(eval_dataset_name)["train"]
    ranking_dataset = load_dataset(ranking_dataset_name)["train"]

    print("Filtering datasets...")
    eval_prompts = set(eval_dataset[text_col])
    decision_ranking_dataset = ranking_dataset.filter(
        lambda x:
        x[best_uid_col] != 'none'
    )

    decision_ranking_dataset = decision_ranking_dataset.filter(
        lambda x:
        x[text_col] not in eval_prompts
    )
    decision_ranking_dataset = filter_dataset_by_text(decision_ranking_dataset)

    df = decision_ranking_dataset.to_pandas()
    df = df.sort_values("created_at").drop_duplicates("prompt")
    df[uid_bad_col] = df.apply(lambda x: random.choice(
        list(set({x["image_1_uid"], x["image_2_uid"], x["image_3_uid"], x["image_4_uid"]}) - set(
            {x[best_uid_col]}))), axis=1)
    df = df[[text_col, uid_bad_col, best_uid_col]]
    df[url_bad_col] = df.apply(
        lambda
            x: f"https://text-to-image-human-preferences.s3.us-east-2.amazonaws.com/images/{x[uid_bad_col]}.png",
        axis=1)
    df[url_good_col] = df.apply(
        lambda
            x: f"https://text-to-image-human-preferences.s3.us-east-2.amazonaws.com/images/{x[best_uid_col]}.png",
        axis=1)

    print("Saving urls for img2dataset...")
    df.to_parquet(urls_out_path)

    if not os.path.exists(images_good_out_path):
        print("Downloading images...")
        download(
            processes_count=process_count,
            thread_count=thread_count,
            url_list=urls_out_path,
            image_size=image_size,
            output_folder=images_good_out_path,
            resize_mode=resize_mode,
            resize_only_if_bigger=True,
            output_format="parquet",
            input_format="parquet",
            url_col=url_good_col,
            caption_col=text_col,
            number_sample_per_shard=num_samples_per_shard,
            distributor="multiprocessing",
            min_image_size=image_size
        )
    else:
        print("Good images already downloaded.")

    if not os.path.exists(images_bad_out_path):
        print("Downloading images...")
        download(
            processes_count=process_count,
            thread_count=thread_count,
            url_list=urls_out_path,
            image_size=image_size,
            output_folder=images_bad_out_path,
            resize_mode=resize_mode,
            resize_only_if_bigger=True,
            output_format="parquet",
            input_format="parquet",
            url_col=url_bad_col,
            caption_col=text_col,
            number_sample_per_shard=num_samples_per_shard,
            distributor="multiprocessing",
            min_image_size=image_size
        )
    else:
        print("Bad images already downloaded.")

    if not os.path.exists(dataset_out_path):
        print("Reading downloaded images...")
        dataset = parquet2dataset(images_good_out_path)
        dataset_bad = parquet2dataset(images_bad_out_path)

        print("Filtering errors...")
        dataset = dataset.filter(lambda x: x['error_message'] is None)
        dataset_bad = dataset_bad.filter(lambda x: x['error_message'] is None)
        prompts_bad = set(dataset_bad['caption'])
        dataset = dataset.filter(lambda x: x['caption'] in prompts_bad)

        with Pool(processes=process_count) as pool:
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

        print(f"saving to disk {dataset_out_path}")
        dataset.save_to_disk(dataset_out_path)
    else:
        print("Dataset already exists.. Reading from disk")
        dataset = load_from_disk(dataset_out_path)

    print(f"splitting dataset to train-test - {ratio=}")
    dataset = dataset.train_test_split(ratio)
    print(f"pushing to hub yuvalkirstain/{dataset_hub_name}")
    dataset.push_to_hub(dataset_hub_name)


if __name__ == '__main__':
    main()
