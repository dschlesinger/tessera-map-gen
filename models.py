"""Shared building blocks for both decoder architectures.

Both decoders take a spatial (128, P, P) Tessera embedding grid as conditioning and
produce a (3, P, P) RGB patch. The embedding is spatial (same resolution as the target
image) so it's conditioned in via channel-concat at the input; the diffusion decoder
additionally conditions each residual block on the diffusion timestep via FiLM.
"""

import math

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> GroupNorm -> SiLU, no conditioning. Used by the one-step decoder."""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SinusoidalTimeEmbedding(nn.Module):
    """Scalar diffusion timestep -> vector, via sinusoidal features + a small MLP."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


class FiLM(nn.Module):
    """Predicts a per-channel (scale, shift) from a conditioning vector."""

    def __init__(self, cond_dim, channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        return x * (1 + scale) + shift


class ResBlock(nn.Module):
    """Residual conv block with optional FiLM conditioning (used for timestep)."""

    def __init__(self, in_ch, out_ch, cond_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.film = FiLM(cond_dim, out_ch) if cond_dim is not None else None
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond=None):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        if self.film is not None and cond is not None:
            h = self.film(h, cond)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class OneStepDecoder(nn.Module):
    """Deterministic embedding -> RGB regressor. No conditioning tricks needed:
    the embedding grid *is* the only input, fed in directly as input channels."""

    def __init__(self, in_ch=128, hidden=128, depth=6):
        super().__init__()
        layers = [ConvBlock(in_ch, hidden)]
        for _ in range(depth - 1):
            layers.append(ConvBlock(hidden, hidden))
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(hidden, 3, 3, padding=1)

    def forward(self, embedding):
        h = self.body(embedding)
        return torch.tanh(self.out(h))


class DiffusionUNet(nn.Module):
    """Small conditional noise-prediction network for the DDPM decoder.

    Input: noisy RGB (3ch) concatenated with the embedding grid (128ch) -> 131ch.
    Conditioning: diffusion timestep, injected via FiLM in every ResBlock.
    No spatial down/upsampling -- patches are small (64x64) and embedding/image share
    resolution, so a stack of same-resolution ResBlocks is enough for this prototype.
    """

    def __init__(self, embedding_dim=128, hidden=128, depth=6, time_dim=128):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(3 + embedding_dim, hidden, 3, padding=1)
        self.blocks = nn.ModuleList(
            [ResBlock(hidden, hidden, cond_dim=time_dim) for _ in range(depth)]
        )
        self.out_norm = nn.GroupNorm(min(8, hidden), hidden)
        self.out_conv = nn.Conv2d(hidden, 3, 3, padding=1)

    def forward(self, noisy_rgb, embedding, t):
        cond = self.time_embed(t)
        h = self.in_conv(torch.cat([noisy_rgb, embedding], dim=1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out_conv(nn.functional.silu(self.out_norm(h)))
