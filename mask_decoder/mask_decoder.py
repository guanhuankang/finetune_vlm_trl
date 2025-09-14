import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from PIL import Image
import numpy as np
import os

from mask_decoder.segment_anything import (
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
)
from mask_decoder.transformer import Transformer

class MaskDecoder(nn.Module):
    def __init__(
        self,
        config,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    ):
        super().__init__()

        self.image_encoder = build_sam_vit_h(
            checkpoint=config.sam_checkpoint
        ).image_encoder

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        width = config.mask_decoder_dim

        self.proj1 = nn.Sequential(
            nn.Linear(256, width),
            nn.ReLU(),
            nn.Linear(width, width)
        )

        self.proj2 = nn.Sequential(
            nn.Linear(config.mask_decoder_proj2_dim, width),
            nn.ReLU(),
            nn.Linear(width, width)
        )

        self.decoder = Transformer(
            n_blocks=config.mask_decoder_n_blocks,
            dim=width,
            num_heads=8,
        )

        self.conv = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
        )

        self.mask_conv = nn.Conv2d(config.n_mask_tokens + 2, 1, 3, padding=1)
    
    def set_trainable(self):
        for name, param in self.named_parameters():
            if "image_encoder" in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    def save_pretrained(self, output_dir):
        """Save only trainable parameters to a file."""
        os.makedirs(output_dir, exist_ok=True)
        trainable_state_dict = {
            name: param.data
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        torch.save(trainable_state_dict, os.path.join(output_dir, "mask_decoder.pth"))
    
    def from_pretrained(self, pretrained_path):
        state_dict = torch.load(
            os.path.join(pretrained_path, "mask_decoder.pth"),
            map_location=torch.device('cpu')  # or 'cuda' if preferred
        )
        self.load_state_dict(state_dict, strict=False)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, x, image_features=None, masks=None, image=None):
        """
        x: B, n, C (mask tokens)
        image_features: 1, c, h, w
        masks: [optional] B, 1, H, W for calc loss
        image: [optional] PIL.Image

        output mask: B, 1, H, W
        """
        image_features = self.image_encoder(self.preprocess(image))  # 1, 256, 64, 64
        image_features = image_features.expand(len(x), -1, -1, -1)   # B, 256, 64, 64

        out = self.decoder(
            torch.cat(
                [
                    self.proj1(rearrange(image_features, "b c h w -> b (h w) c")),
                    self.proj2(x),
                ],
                dim=1,
            )
        )

        W, H = image.size if not isinstance(image, type(None)) else (640, 480)
        h, w = image_features.shape[-2::]

        x = out[:, int(h*w)::, :] ## B, n, C
        
        image_features = rearrange(out[:, 0 : int(h * w), :], "b (h w) c -> b c h w", h=h, w=w)
        image_features = nn.Sequential(nn.Upsample(size=(256, 256), mode="bilinear"), self.conv)(image_features)

        m = einsum(x, image_features, "b k c, b c h w -> b k h w")
        m = self.mask_conv(m) # b, 1, h, w
        
        if masks != None:
            m = F.interpolate(m, size=masks.shape[2::], mode="bilinear")
            loss = F.binary_cross_entropy_with_logits(m, masks.float())
        else:
            m = F.interpolate(m, size=(H, W), mode="bilinear")
            loss = None
        
        return {
            "masks": m.sigmoid(),
            "loss": loss
        }

    def preprocess(self, x: Image) -> torch.Tensor:
        """return 1, 3, 1024, 1024"""
        a = self.image_encoder.img_size
        x = torch.tensor(np.array(x.resize((a, a), Image.BILINEAR)), device=self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).float()
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        # h, w = x.shape[-2:]
        # padh = self.image_encoder.img_size - h
        # padw = self.image_encoder.img_size - w
        # x = F.pad(x, (0, padw, 0, padh))
        return x


if __name__ == "__main__":
    from config import PSORConfig

    config = PSORConfig()

    md = MaskDecoder(config).cuda()
    print(md)
    # for name, param in md.named_parameters():
    #     print(name)

    image = Image.fromarray((np.random.random((480, 640, 3)) * 255).astype(np.uint8))
    x = torch.rand(3, 6, 2048).cuda()
    f = torch.rand(1, 2048, 74, 74).cuda()

    y = md(x, f, masks=torch.rand(3, 1, 480, 640).gt(0.5).float().cuda(), image=image)

    # print(y)
    print(y["masks"].shape)
