import torch
from torchvision import transforms
from utils import convert_to_RGB


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, image_processor, caption_column_name, image_column_name, bad_image_column_name):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = image_processor
        self.caption_column_name = caption_column_name
        self.image_column_name = image_column_name
        self.bad_image_column_name = bad_image_column_name
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __len__(self):
        return len(self.dataset)

    def tokenize(self, example):
        caption = example[self.caption_column_name]
        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        return input_ids

    def __getitem__(self, idx):
        example = self.dataset[idx]
        input_ids = self.tokenize(example)
        image = example[self.image_column_name]
        pixel_values = self.transform(image, return_tensors="pt")["pixel_values"]
        bad_image = example[self.bad_image_column_name]
        bad_pixel_values = self.transform(bad_image, return_tensors="pt")["pixel_values"]
        item = {"input_ids": input_ids, "pixel_values": pixel_values, "bad_pixel_values": bad_pixel_values}
        return item

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        bad_pixel_values = torch.cat([item["bad_pixel_values"] for item in batch], dim=0)
        bad_pixel_values = bad_pixel_values.to(memory_format=torch.contiguous_format).float()
        collated = {"input_ids": input_ids, "pixel_values": pixel_values, "bad_pixel_values": bad_pixel_values}
        return collated
