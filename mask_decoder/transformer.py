import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
from einops import rearrange

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: B, N, C"""
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = rearrange(self.qkv(x), "b n (n3 nh c) -> n3 (b nh) n c", n3=3, nh=self.num_heads)
        q, k, v = qkv.unbind(0) ## (b nh) n c
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = rearrange(x, "(b nh) n c -> b n (nh c)", nh=self.num_heads)
        x = self.proj(x)
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
    
if __name__=="__main__":
    b, n, width = 1, 3, 32
    m = Transformer(n_blocks=2, dim=width, num_heads=4).cuda()
    x = torch.rand(b, n, width).cuda()
    y = m(x)

    print(m)
    print(x.shape, y.shape)
