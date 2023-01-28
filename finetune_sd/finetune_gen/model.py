import torch
import einops


class Model(torch.nn.Module):

    def __init__(
            self,
            unet,
            text_encoder,
            vae,
            unconditional_input_ids,
            classifier_free_ratio,
            noise_scheduler,
            use_pixel_loss,
            weight_dtype
    ):
        super().__init__()
        self.unet = unet
        self.text_encoder = text_encoder
        self.vae = vae
        self.unconditional_input_ids = unconditional_input_ids
        self.classifier_free_ratio = classifier_free_ratio
        self.noise_scheduler = noise_scheduler
        self.use_pixel_loss = use_pixel_loss
        self.weight_dtype = weight_dtype

    def get_image_pred(self, model_output, timestep, sample):

        t = timestep

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)

        # 1. compute alphas, betas
        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t][:, None, None, None]
        beta_prod_t = 1 - alpha_prod_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.noise_scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.noise_scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.noise_scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip "predicted x_0"
        if self.noise_scheduler.config.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        return pred_original_sample

    @torch.no_grad()
    def encode_image(self, image):
        latents = self.vae.encode(image.to(self.weight_dtype)).latent_dist.sample()
        latents = latents * 0.18215
        return latents

    def decode_latents(self, latents):
        latents = (1 / 0.18215) * latents
        image = self.vae.decode(latents).sample
        return image

    def forward(self, pixel_values, input_ids, face_mask=None):

        with torch.no_grad():
            # classifier free
            classifier_free_mask = torch.rand(size=(input_ids.shape[0],)) < self.classifier_free_ratio
            input_ids[classifier_free_mask] = self.unconditional_input_ids.to(input_ids.device)

            # encode image into latents
            latents = self.encode_image(pixel_values)
            latents = latents.float()

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep (forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # get text condition
        encoder_hidden_states = self.text_encoder(input_ids)[0]

        # run unet
        model_pred = self.unet(noisy_latents.to(self.weight_dtype), timesteps, encoder_hidden_states).sample

        if self.use_pixel_loss:
            target = pixel_values.clone()
            latent_pred = self.get_image_pred(model_pred, timesteps, noisy_latents)
            image_pred = self.decode_latents(latent_pred.to(self.weight_dtype))
            image_pred = torch.clamp(image_pred, -1, 1)
            if face_mask is not None:
                image_pred_face = image_pred * face_mask
                target_face = target * face_mask

                face_loss = torch.nn.functional.mse_loss(image_pred_face.float(), target_face.float(), reduction="none")
                face_loss = einops.rearrange(face_loss, "b c h w -> b (c h w)")
                face_mask = einops.rearrange(face_mask, "b c h w -> b (c h w)")
                face_loss = face_loss.sum(dim=-1) / (face_mask.sum(dim=-1) + 1e-8)
                face_loss = face_loss.mean()

                reg_loss = torch.nn.functional.mse_loss(image_pred.float(), target.float(), reduction="mean")

                loss = (face_loss * 4 + reg_loss) / 5
            else:
                loss = torch.nn.functional.mse_loss(image_pred.float(), target.float(), reduction="mean")
        else:
            loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    def enable_gradient_checkpointing(self):
        self.unet.enable_gradient_checkpointing()
        self.text_encoder.gradient_checkpointing_enable()
