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
from mask_decoder.unet import UNet

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

        width = 256

        self.unet = UNet(width=width)

        self.mask_weight = nn.Parameter(torch.randn((1, width)), requires_grad=True)

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    
    def set_trainable(self):
        for name, param in self.named_parameters():
            if "image_encoder" in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    def save_pretrained(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(output_dir, "mask_decoder.pth"))
    
    def from_pretrained(self, pretrained_path):
        state_dict = torch.load(
            os.path.join(pretrained_path, "mask_decoder.pth"),
            map_location=torch.device('cuda')
        )
        self.load_state_dict(state_dict, strict=True)

    @property
    def device(self):
        return self.pixel_mean.device

    def bbox_to_mask(self, bbox, H=64, W=64):
        x1, y1, x2, y2 = bbox.toint()
        m = torch.zeros((H, W), device=self.device)
        m[y1:y2+1, x1:x2+1] = 1.0
        return m[None, None, :, :] * self.mask_weight[:, :, None, None]
    
    def forward(self, records, image, masks=None):
        """
        records: list of dict K*
            {
                "rank": int, // 1, 2, 3, ...
                "category": str,
                "bbox": BBox,
            }
        image: PIL image
        masks: None or Tensor(K*, 1, H, W)

        output mask: K*, 1, H, W
        """
        image_features = self.image_encoder(self.preprocess(image))  # 1, 256, 64, 64
        image_features = image_features.expand(len(records), -1, -1, -1)   # B, 256, 64, 64
        _, _, H, W = image_features.shape

        bbox_promt = torch.cat([self.bbox_to_mask(r["bbox"], H=H, W=W) for r in records], dim=0)
        image_features = image_features + bbox_promt
        
        m = self.unet(image_features)
        m = F.interpolate(m, size=tuple(reversed(image.size)), mode="bilinear")
        
        if masks != None:
            loss = F.binary_cross_entropy_with_logits(
                F.interpolate(m, size=masks.shape[2::], mode="bilinear"), 
                masks.float()
            )
        else:
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
    from utils import BBox

    config = PSORConfig()

    md = MaskDecoder(config).cuda()
    print(md)
    
    B = 3
    image = Image.fromarray((np.random.random((480, 640, 3)) * 255).astype(np.uint8))
    masks=torch.rand(B, 1, 480, 640).gt(0.5).float().cuda()
    records = [{
        "bbox": BBox({
            "x1": 10,
            "y1": 10,
            "x2": 23,
            "y2": 63
        })
    }]*B

    y = md(records, image, masks=masks)

    print(y["masks"].shape, y["loss"])
