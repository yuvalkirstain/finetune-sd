import torch

try:
    import torch.distributed.nn
    has_distributed = True
except ImportError:
    has_distributed = False


class Model(torch.nn.Module):

    def __init__(
            self,
            clip_model,
            accelerator,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.accelerator = accelerator

    def gather_features(self, features):
        assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        return all_features

    def calc_loss(self, image_features, bad_image_features, text_features, logit_scale):
        device = image_features.device
        if self.accelerator.num_processes > 1:
            image_features = self.gather_features(image_features)
            bad_image_features = self.gather_features(bad_image_features)
            text_features = self.gather_features(text_features)
        all_image_features = torch.cat([image_features, bad_image_features], dim=0)  # (2 * batch_size, dim)
        # all_image_features = image_features
        logits_per_image = logit_scale * all_image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ all_image_features.T
        num_logits = logits_per_text.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        good_images_logits = logits_per_image.chunk(2, dim=0)[0]
        image_loss = torch.nn.functional.cross_entropy(good_images_logits, labels)
        # image_loss = torch.nn.functional.cross_entropy(logits_per_image, torch.cat([labels, labels], dim=0))
        text_loss = torch.nn.functional.cross_entropy(logits_per_text, labels)
        loss = (image_loss + text_loss) / 2
        return loss

    def forward(self, pixel_values, input_ids, bad_pixel_values):
        all_pixel_values = torch.cat([pixel_values, bad_pixel_values], dim=0)
        all_image_features = self.clip_model.get_image_features(all_pixel_values)
        text_features = self.clip_model.get_text_features(input_ids)
        all_image_features = all_image_features / all_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features, bad_image_features = all_image_features.chunk(2, dim=0)
        loss = self.calc_loss(image_features, bad_image_features, text_features, self.clip_model.logit_scale.exp())
        return loss

    def enable_gradient_checkpointing(self):
        self.clip_model.gradient_checkpointing_enable()
