import os.path

from datasets import load_dataset
from transformers import BlipForConditionalGeneration, AutoProcessor, GenerationConfig
import torch


def generate_caption(processor, model, image, generation_config, device, dtype):
    inputs = processor(images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(pixel_values=inputs.pixel_values.to(dtype), generation_config=generation_config)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_caption


def main():
    dataset_name = "pexel_friends"
    dataset_full_name = f"yuvalkirstain/{dataset_name}"
    model_id = "Salesforce/blip-image-captioning-large"
    out_name = f"{dataset_name}_with_generated_captions"
    if os.path.exists(out_name):
        print(f"Dataset {out_name} already exists, skipping")
        dataset = load_dataset(out_name)
        dataset.push_to_hub(out_name)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    generation_config = GenerationConfig(
        max_length=50,
        num_beams=4,
        repetition_penalty=1.4
    )

    print("Loading dataset...")
    dataset = load_dataset(dataset_full_name)
    print(dataset)

    print("Loading model...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    model = model.to(dtype=dtype)
    model = model.to(device)
    model.eval()

    for split in ["train", "test"]:
        if split not in dataset:
            continue
        dataset_split = dataset[split]
        print(f"Generating captions for {split} split...")
        captions = []
        for i, e in enumerate(dataset_split):
            image = e["image"]
            caption = generate_caption(processor, model, image, generation_config, device, dtype)
            # if "portrait" not in caption.lower():
            #     caption = f"Portrait, {caption}"
            print(f"===================  {i}/{len(dataset_split)}  =====================")
            print(f"Original Caption: {e['text']}")
            print(f"Caption: {caption}")
            captions.append(caption)

        dataset_split = dataset_split.add_column("generated_caption", captions)
        dataset[split] = dataset_split
    dataset.save_to_disk(out_name)
    dataset.push_to_hub(out_name)




if __name__ == '__main__':
    main()
