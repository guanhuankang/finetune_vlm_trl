import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from mask_decoder.transformer import Transformer

class UNet(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        w, w2, w4 = width, width * 2, width * 4

        self.down_block_1 = DownBlock(w, w2)
        self.down_block_2 = DownBlock(w2, w4)

        self.bottle_neck = SpatialTransformer(n_blocks=2, dim=w4, num_heads=w4//16)

        self.up_block_1 = UpBlock(w4, w2)
        self.up_block_2 = UpBlock(w2, w)

        self.up_cls = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(w, w//2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(w//2, w//4),
            nn.Conv2d(w//4, 1, 1),
        )

    def forward(self, x, prompt):
        x = self.up_block_2(
                self.up_block_1(
                    self.bottle_neck(
                        self.down_block_2(
                            self.down_block_1(x + prompt)
                        )
                    ),
                    self.down_block_1(x)
                ),
                x
        )
        
        x = self.up_cls(x)

        return x

class SpatialTransformer(nn.Module):
    def __init__(self, n_blocks, dim, num_heads):
        super().__init__()

        self.transformer = Transformer(
            n_blocks=n_blocks,
            dim=dim,
            num_heads=num_heads
        )
    
    def forward(self, x):
        _, _, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.transformer(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv_block(x)

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv_block = ConvBlock(in_c, out_c)
        self.down_layer = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv_block(x)
        x = self.down_layer(x)
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up_layer = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv_shrink = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv_block = ConvBlock(out_c * 2, out_c)

    def forward(self, x, skip):
        x = self.up_layer(x)
        x = self.conv_shrink(x)
        x = self.conv_block(torch.cat([x, skip], dim=1))
        return x

if __name__=="__main__":
    unet = UNet(width=256).cuda()
    x = torch.rand(8, 256, 64, 64).cuda()
    y = unet(x)
    print(x.shape, y.shape)