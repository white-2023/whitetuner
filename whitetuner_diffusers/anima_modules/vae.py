"""
Anima VAE - Wan 2.1 VAE 实现

用于图像编解码，16通道3D latent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, List
from safetensors.torch import load_file


CACHE_T = 2


class Wan21LatentFormat:
    latent_channels = 16
    latent_dimensions = 3
    scale_factor = 1.0
    
    def __init__(self):
        self.latents_mean = torch.tensor([
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]).view(1, self.latent_channels, 1, 1, 1)
        self.latents_std = torch.tensor([
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]).view(1, self.latent_channels, 1, 1, 1)
    
    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std
    
    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x, cache_x=None):
        kernel_t = self.weight.shape[2]
        t = x.shape[2]
        
        pad_needed = max(self._padding, kernel_t - t)
        if pad_needed > 0:
            padding_shape = list(x.shape)
            padding_shape[2] = pad_needed
            padding = torch.zeros(padding_shape, device=x.device, dtype=x.dtype)
            x = torch.cat([padding, x], dim=2)
        
        return super().forward(x)


class RMSNorm3d(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        norm = F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma.to(x)
        if self.bias is not None:
            norm = norm + self.bias.to(x)
        return norm


class Resample(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                nn.Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x):
        b, c, t, h, w = x.size()
        
        if self.mode == 'upsample3d' and t > 1:
            x = self.time_conv(x)
            x = x.reshape(b, 2, c, t, h, w)
            x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
            x = x.reshape(b, c, t * 2, h, w)
        
        t_curr = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t_curr)

        if self.mode == 'downsample3d' and t > 1:
            x = self.time_conv(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMSNorm3d(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMSNorm3d(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = RMSNorm3d(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        
        q = q.reshape(b * t, c, -1).transpose(1, 2)
        k = k.reshape(b * t, c, -1).transpose(1, 2)
        v = v.reshape(b * t, c, -1).transpose(1, 2)
        
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b * t, c, h, w)
        
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        input_channels=3,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0
    ):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv1 = CausalConv3d(input_channels, dims[0], 3, padding=1)

        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.ModuleList(downsamples)

        self.middle = nn.ModuleList([
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout)
        ])

        self.head = nn.Sequential(
            RMSNorm3d(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x)
        for layer in self.middle:
            x = layer(x)
        x = self.head(x)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim=128,
        z_dim=4,
        output_channels=3,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0
    ):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        self.middle = nn.ModuleList([
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout)
        ])

        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.ModuleList(upsamples)

        self.head = nn.Sequential(
            RMSNorm3d(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, output_channels, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.upsamples:
            x = layer(x)
        x = self.head(x)
        return x


class WanVAE(nn.Module):
    def __init__(
        self,
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        image_channels=3,
        dropout=0.0
    ):
        super().__init__()
        self.z_dim = z_dim
        self.temperal_upsample = temperal_downsample[::-1]

        self.encoder = Encoder3d(
            dim, z_dim * 2, image_channels, dim_mult, 
            num_res_blocks, attn_scales, temperal_downsample, dropout
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dim, z_dim, image_channels, dim_mult, 
            num_res_blocks, attn_scales, self.temperal_upsample, dropout
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.conv1(h)
        mu, log_var = h.chunk(2, dim=1)
        return mu

    def decode(self, z):
        z = self.conv2(z)
        return self.decoder(z)


class AnimaVAE:
    def __init__(
        self,
        vae_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype
        self.latent_format = Wan21LatentFormat()
        
        print(f"Loading VAE from {vae_path}")
        
        state_dict = load_file(vae_path)
        
        dim = state_dict["decoder.head.0.gamma"].shape[0]
        z_dim = 16
        image_channels = state_dict["encoder.conv1.weight"].shape[1]
        
        print(f"  - Detected config: dim={dim}, z_dim={z_dim}, image_channels={image_channels}")
        
        self.model = WanVAE(
            dim=dim,
            z_dim=z_dim,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            attn_scales=[],
            temperal_downsample=[False, True, True],
            image_channels=image_channels,
            dropout=0.0
        )
        
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  - Missing keys: {len(missing)}")
        if unexpected:
            print(f"  - Unexpected keys: {len(unexpected)}")
        
        self.model = self.model.to(device=device, dtype=dtype)
        self.model.eval()
        print("VAE loaded successfully")

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def encode(self, images: List[torch.Tensor]) -> List[torch.Tensor]:
        results = []
        for img in images:
            if img.dim() == 4:
                img = img.unsqueeze(0)
            img = img.to(self.device, self.dtype)
            latent = self.model.encode(img)
            latent = self.latent_format.process_in(latent)
            results.append(latent.squeeze(0))
        return results

    @torch.no_grad()
    def decode(self, latents: List[torch.Tensor], already_processed: bool = False) -> List[torch.Tensor]:
        results = []
        for lat in latents:
            if lat.dim() == 4:
                lat = lat.unsqueeze(0)
            lat = lat.to(self.device, self.dtype)
            if not already_processed:
                lat = self.latent_format.process_out(lat)
            img = self.model.decode(lat)
            img = img.clamp(-1, 1)
            results.append(img.squeeze(0))
        return results


def load_anima_vae(
    vae_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> AnimaVAE:
    return AnimaVAE(vae_path, device, dtype)
