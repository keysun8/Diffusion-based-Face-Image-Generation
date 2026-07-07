import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from diffusers import AutoencoderKL

#  Args 
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint (.pt)")
parser.add_argument("--output_dir", type=str, default="inference_outputs", help="Where to save generated images")
args = parser.parse_args()

#  Configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

T = 1000
IMG_SIZE = 32          # latent spatial dim: 256px / 8 (VAE) = 32
CHANNELS = 4           # SD VAE latent channels
TIME_DIM = 320         # sinusoidal embedding dim
TIME_EMB_DIM = 1280    # projected time embedding dim

OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#  DDPM Noise Schedule 
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bars)
sqrt_recip_a = torch.sqrt(1.0 / alphas)

betas_tilde = betas.clone()
betas_tilde[1:] = betas[1:] * (1.0 - alpha_bars[:-1]) / (1.0 - alpha_bars[1:])


# Model Architecture Components 
class TimeEmbedding(nn.Module):
    def __init__(self, dim=TIME_DIM, time_dim=TIME_EMB_DIM):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        args_ = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.sin(args_), torch.cos(args_)], dim=1)
        return self.mlp(emb)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim=TIME_EMB_DIM):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x, time_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch, n_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.qkv = nn.Linear(ch, 3 * ch)
        self.out = nn.Linear(ch, ch)
        self.n_heads = n_heads
        self.d_head = ch // n_heads

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)
        qkv = self.qkv(h).chunk(3, dim=-1)
        q, k, v = [t.view(B, H * W, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1) / scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.out(out).transpose(1, 2).view(B, C, H, W)
        return x + out


class AttentionBlock(nn.Module):
    def __init__(self, ch, n_heads=8):
        super().__init__()
        self.attn = SelfAttention(ch, n_heads)
        self.norm = nn.LayerNorm(ch)
        self.ff1 = nn.Linear(ch, 4 * ch * 2)
        self.ff2 = nn.Linear(4 * ch, ch)

    def forward(self, x):
        x = self.attn(x)
        B, C, H, W = x.shape
        h = x.view(B, C, H * W).transpose(1, 2)
        gate, val = self.ff1(self.norm(h)).chunk(2, dim=-1)
        h = self.ff2(gate * F.gelu(val)) + h
        return h.transpose(1, 2).view(B, C, H, W)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, in_ch=CHANNELS, base_ch=128):
        super().__init__()
        ch = base_ch

        self.stem = nn.Conv2d(in_ch, ch, 3, padding=1)

        self.enc0_r1 = ResidualBlock(ch, ch)
        self.enc0_r2 = ResidualBlock(ch, ch)
        self.enc0_dn = Downsample(ch)

        self.enc1_r1 = ResidualBlock(ch, ch * 2)
        self.enc1_a1 = AttentionBlock(ch * 2)
        self.enc1_r2 = ResidualBlock(ch * 2, ch * 2)
        self.enc1_a2 = AttentionBlock(ch * 2)
        self.enc1_dn = Downsample(ch * 2)
        ch2 = ch * 2

        self.enc2_r1 = ResidualBlock(ch2, ch2 * 2)
        self.enc2_a1 = AttentionBlock(ch2 * 2)
        self.enc2_r2 = ResidualBlock(ch2 * 2, ch2 * 2)
        self.enc2_a2 = AttentionBlock(ch2 * 2)
        self.enc2_dn = Downsample(ch2 * 2)
        ch4 = ch2 * 2

        self.enc3_r1 = ResidualBlock(ch4, ch4)
        self.enc3_a1 = AttentionBlock(ch4)
        self.enc3_r2 = ResidualBlock(ch4, ch4)
        self.enc3_a2 = AttentionBlock(ch4)

        self.mid_r1 = ResidualBlock(ch4, ch4)
        self.mid_a = AttentionBlock(ch4)
        self.mid_r2 = ResidualBlock(ch4, ch4)

        self.dec3_r1 = ResidualBlock(ch4 * 2, ch4)
        self.dec3_a1 = AttentionBlock(ch4)
        self.dec3_r2 = ResidualBlock(ch4 * 2, ch4)
        self.dec3_a2 = AttentionBlock(ch4)
        self.dec3_r3 = ResidualBlock(ch4 * 2, ch4)
        self.dec3_a3 = AttentionBlock(ch4)
        self.dec3_up = Upsample(ch4)

        self.dec2_r1 = ResidualBlock(ch4 + ch4, ch4)
        self.dec2_a1 = AttentionBlock(ch4)
        self.dec2_r2 = ResidualBlock(ch4 + ch4, ch2)
        self.dec2_a2 = AttentionBlock(ch2)
        self.dec2_r3 = ResidualBlock(ch2 + ch2, ch2)
        self.dec2_up = Upsample(ch2)

        self.dec1_r1 = ResidualBlock(ch2 + ch2, ch2)
        self.dec1_a1 = AttentionBlock(ch2)
        self.dec1_r2 = ResidualBlock(ch2 + ch2, ch)
        self.dec1_a2 = AttentionBlock(ch)
        self.dec1_r3 = ResidualBlock(ch + ch, ch)
        self.dec1_up = Upsample(ch)

        self.dec0_r1 = ResidualBlock(ch + ch, ch)
        self.dec0_r2 = ResidualBlock(ch + ch, ch)

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_ch, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, temb):
        h = self.stem(x)

        s00 = self.enc0_r1(h, temb); s01 = self.enc0_r2(s00, temb)
        h = self.enc0_dn(s01); s0d = h

        h = self.enc1_r1(h, temb); s10 = self.enc1_a1(h)
        h = self.enc1_r2(s10, temb); s11 = self.enc1_a2(h)
        h = self.enc1_dn(s11); s1d = h

        h = self.enc2_r1(h, temb); s20 = self.enc2_a1(h)
        h = self.enc2_r2(s20, temb); s21 = self.enc2_a2(h)
        h = self.enc2_dn(s21); s2d = h

        h = self.enc3_r1(h, temb); s30 = self.enc3_a1(h)
        h = self.enc3_r2(s30, temb); s31 = self.enc3_a2(h)

        h = self.mid_r1(h, temb)
        h = self.mid_a(h)
        h = self.mid_r2(h, temb)

        h = self.dec3_r1(torch.cat([h, s31], dim=1), temb); h = self.dec3_a1(h)
        h = self.dec3_r2(torch.cat([h, s30], dim=1), temb); h = self.dec3_a2(h)
        h = self.dec3_r3(torch.cat([h, s2d], dim=1), temb); h = self.dec3_a3(h)
        h = self.dec3_up(h)

        h = self.dec2_r1(torch.cat([h, s21], dim=1), temb); h = self.dec2_a1(h)
        h = self.dec2_r2(torch.cat([h, s20], dim=1), temb); h = self.dec2_a2(h)
        h = self.dec2_r3(torch.cat([h, s1d], dim=1), temb)
        h = self.dec2_up(h)

        h = self.dec1_r1(torch.cat([h, s11], dim=1), temb); h = self.dec1_a1(h)
        h = self.dec1_r2(torch.cat([h, s10], dim=1), temb); h = self.dec1_a2(h)
        h = self.dec1_r3(torch.cat([h, s0d], dim=1), temb)
        h = self.dec1_up(h)

        h = self.dec0_r1(torch.cat([h, s01], dim=1), temb)
        h = self.dec0_r2(torch.cat([h, s00], dim=1), temb)

        return self.out(h)


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(TIME_DIM, TIME_EMB_DIM)
        self.unet = UNet()

    def forward(self, x, t):
        temb = self.time_embedding(t)
        return self.unet(x, temb)


#  Setup Models 
print("Loading VAE decoder...")
vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae"
).to(device).eval()

for p in vae.parameters():
    p.requires_grad_(False)

LATENT_SCALE = vae.config.scaling_factor  # read from VAE config instead of hardcoding

print(f"Loading Diffusion model from {args.ckpt}...")
model = Diffusion().to(device)
checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model"])
model.eval()

use_amp = device.type == "cuda"  # autocast only makes sense on GPU


#  Inference Functions 
@torch.no_grad()
def decode_latents(latents):
    latents = latents.to(device) / LATENT_SCALE
    with torch.amp.autocast("cuda", enabled=use_amp):
        imgs = vae.decode(latents).sample
    return (imgs / 2 + 0.5).clamp(0, 1)


@torch.no_grad()
def generate_single_image():
    print("Starting reverse diffusion for 1 sample...")
    x = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE, device=device)

    for t_val in reversed(range(T)):
        t_batch = torch.full((1,), t_val, device=device, dtype=torch.long)

        with torch.amp.autocast("cuda", enabled=use_amp):
            eps = model(x, t_batch)

        coef = (1.0 - alphas[t_val]) / sqrt_one_minus_ab[t_val]
        mean = sqrt_recip_a[t_val] * (x - coef * eps)

        if t_val > 0:
            noise = torch.randn_like(x)
            x = mean + (torch.sqrt(betas_tilde[t_val]) * noise)
        else:
            x = mean

        if t_val % 100 == 0 or t_val == T - 1:
            print(f"Step {T - t_val}/{T} complete.")

    print("Decoding latents to pixel space...")
    return decode_latents(x)


#  Execution 
if __name__ == "__main__":
    final_image = generate_single_image()

    r = torch.randint(2001, 3000, (1,)).item()  # random suffix for filename uniqueness
    save_path = OUTPUT_DIR / f"generated_sample_{r}.png"
    vutils.save_image(final_image, str(save_path))

    print(f"Success! Output tensor shape: {final_image.shape}")
    print(f"Image saved to {save_path}")
