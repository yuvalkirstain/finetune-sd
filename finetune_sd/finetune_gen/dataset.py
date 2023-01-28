import torch
from torchvision import transforms
import mediapipe as mp
from utils import convert_to_RGB
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, resolution, add_face_mask, caption_column_name, image_column_name):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                convert_to_RGB,
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2. - 1.)
            ]
        )
        self.add_face_mask = add_face_mask
        self.caption_column_name = caption_column_name
        self.image_column_name = image_column_name
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

    def mask_face(self, image):
        mask = torch.zeros_like(self.to_tensor(image))
        with mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5) as face_detection:
            image_cv = np.asarray(image)
            results = face_detection.process(image_cv)
            if results.detections is not None:
                image_hight, image_width, _ = image_cv.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x_min = int(bbox.xmin * image_hight)
                    y_min = int(bbox.ymin * image_hight)
                    x_max = x_min + int(bbox.width * image_width)
                    y_max = y_min + int(bbox.height * image_hight)
                    mask[:, y_min:y_max, x_min:x_max] = 1
            else:
                mask.fill_(1)
        return mask

    def mask_hands(self, image):
        mask = torch.zeros_like(self.to_tensor(image))
        flipped_image = transforms.functional.hflip(image)
        image_cv = np.asarray(flipped_image)
        with mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.7
        ) as hands:
            results = hands.process(image_cv)
            if results.multi_hand_landmarks is not None:
                image_hight, image_width, _ = image_cv.shape
                mask = transforms.functional.hflip(mask)

                for hand_landmarks in results.multi_hand_landmarks:
                    x_max = 0
                    y_max = 0
                    x_min = image_width
                    y_min = image_hight
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_hight)
                        x_max = max(x_max, x)
                        y_max = max(y_max, y)
                        x_min = min(x_min, x)
                        y_min = min(y_min, y)
                    mask[:, y_min:y_max, x_min:x_max] = 1
                mask = transforms.functional.hflip(mask)
        return mask

    def __getitem__(self, idx):
        example = self.dataset[idx]
        input_ids = self.tokenize(example)
        image = example[self.image_column_name]
        pixel_values = self.transform(image)
        item = {"input_ids": input_ids, "pixel_values": pixel_values}
        if self.add_face_mask:
            post_image_t = pixel_values * 0.5 + 0.5  # canceling the normalization
            post_image = self.to_pil(post_image_t)
            face_mask = self.mask_face(post_image)
            hand_mask = self.mask_hands(post_image)
            item["face_mask"] = torch.logical_or(hand_mask, face_mask).float()
        return item

    @staticmethod
    def collate_fn(batch):
        input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        collated = {"input_ids": input_ids, "pixel_values": pixel_values}
        if "face_mask" in batch[0]:
            face_mask = torch.stack([item["face_mask"] for item in batch])
            face_mask = face_mask.to(memory_format=torch.contiguous_format).float()
            collated["face_mask"] = face_mask
        return collated
