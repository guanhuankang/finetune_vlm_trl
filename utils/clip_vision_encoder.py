from transformers import CLIPVisionModel
import torch
import torch.nn as nn

class ClipVisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.clip = CLIPVisionModel.from_pretrained(config.clip_vision_model)
        
        self.image_size = config.clip_vision_model_image_size
        
        self.register_buffer("pixel_mean", torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False)

    def forward(self, batched_input):
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.clip(input_images).last_hidden_state[:, 1:, :]
        return image_embeddings

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and resize to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Resize
        x = nn.functional.interpolate(x.unsqueeze(0), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)[0]
        return x