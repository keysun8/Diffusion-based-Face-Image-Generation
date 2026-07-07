import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from PIL import Image
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from tqdm import tqdm

#  Args 
parser = argparse.ArgumentParser()
parser.add_argument("--data_root_pt", type=str, required=True, help="Directory of pre-encoded VAE latents (.pt files)")
parser.add_argument("--data_root_jpg", type=str, required=True, help="Directory of real RGB images, used for FID reference stats")
parser.add_argument("--ckpt_dir", type=str, default="checkpoints", help="Where to save model checkpoints")
parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to a checkpoint (.pt) to resume training from. Leave unset to train from scratch")
parser.add_argument("--sample_dir", type=str, default="samples", help="Where to save sample image grids during training")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
parser.add_argument("--log_loss_every", type=int, default=100)
parser.add_argument("--save_img_every", type=int, default=1500)
parser.add_argument("--calc_fid_every", type=int, default=4500)
parser.add_argument("--save_ckpt_every_epoch", type=int, default=5)
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)

#  Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
use_amp = device.type == "cuda"  # autocast only makes sense on GPU

#  Hyperparams 
T = 1000
IMG_SIZE = 32          # latent spatial dim: 256px / 8 (VAE) = 32
CHANNELS = 4           # SD VAE latent channels
TIME_DIM = 320         # sinusoidal embedding dim
TIME_EMB_DIM = 1280    # projected time embedding dim

DATA_ROOT_PT = args.data_root_pt
DATA_ROOT_JPG = args.data_root_jpg
CKPT_DIR = args.ckpt_dir
SAMPLE_DIR = args.sample_dir

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

#  DDPM noise schedule 
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)
sqrt_ab = torch.sqrt(alpha_bars)
sqrt_one_minus_ab = torch.sqrt(1.0 - alpha_bars)
sqrt_recip_a = torch.sqrt(1.0 / alphas)

betas_tilde = betas.clone()
betas_tilde[1:] = betas[1:] * (1.0 - alpha_bars[:-1]) / (1.0 - alpha_bars[1:])


#  Forward diffusion 
def q_sample(x0, t):
    noise = torch.randn_like(x0)
    xt = (sqrt_ab[t, None, None, None] * x0
          + sqrt_one_minus_ab[t, None, None, None] * noise)
    return xt, noise


#  VAE (decode only, frozen) 
print("Loading VAE...")
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
vae.to(device).eval()
for p in vae.parameters():
    p.requires_grad_(False)
SCALING_FACTOR = vae.config.scaling_factor  # 0.18215


#  Datasets & Loaders 
class LatentDataset(Dataset):
    """
    Recursively finds all .pt files under latent_dir.
    Returns float32 tensors of shape (4, 32, 32).
    """
    def __init__(self, latent_dir, unscale: bool = False):
        self.latent_dir = Path(latent_dir)
        self.unscale = unscale
        self.files = sorted(self.latent_dir.rglob("*.pt"))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .pt files found under {latent_dir}")

        print(f"LatentDataset: {len(self.files):,} latents from {latent_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        latent = torch.load(self.files[idx], map_location="cpu", weights_only=True)
        latent = latent.float()  # saved as fp16 to save disk
        if self.unscale:
            latent = latent / SCALING_FACTOR
        return latent


class JpgDataset(Dataset):
    """Loads raw RGB images for the real data distribution in FID."""
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.rglob("*.jpg")) + sorted(self.root_dir.rglob("*.png"))

        if len(self.files) == 0:
            print(f"[WARN] No images found in {root_dir} for FID calculation.")

        # Torchmetrics FID with normalize=True expects float tensors in [0, 1]
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


def make_latent_loader(latent_dir, batch_size, num_workers=4):
    """Creates the DataLoader for the pre-encoded VAE latents."""
    dataset = LatentDataset(latent_dir, unscale=False)  # train on scaled latents (N(0,1))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# Time Embedding 
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
        emb = torch.cat([torch.sin(args_), torch.cos(args_)], dim=1)  # (B, dim)
        return self.mlp(emb)  # (B, time_dim)


# ── UNet Blocks ───────────────────────────────────────────────────────────────
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
    """Multi-head self-attention over spatial tokens."""
    def __init__(self, ch, n_heads=8):
        super().__init__()
        assert ch % n_heads == 0, f"ch={ch} not divisible by n_heads={n_heads}"
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.qkv = nn.Linear(ch, 3 * ch)
        self.out = nn.Linear(ch, ch)
        self.n_heads = n_heads
        self.d_head = ch // n_heads

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        qkv = self.qkv(h).chunk(3, dim=-1)
        q, k, v = [t.view(B, H * W, self.n_heads, self.d_head).transpose(1, 2) for t in qkv]

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1) / scale).softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, H * W, C)
        out = self.out(out).transpose(1, 2).view(B, C, H, W)
        return x + out


class AttentionBlock(nn.Module):
    """ResBlock + SelfAttention + feedforward (GEGLU)."""
    def __init__(self, ch, n_heads=8):
        super().__init__()
        self.attn = SelfAttention(ch, n_heads)
        self.norm = nn.LayerNorm(ch)
        self.ff1 = nn.Linear(ch, 4 * ch * 2)  # GEGLU gate
        self.ff2 = nn.Linear(4 * ch, ch)

    def forward(self, x):
        x = self.attn(x)
        B, C, H, W = x.shape
        h = x.view(B, C, H * W).transpose(1, 2)  # (B, HW, C)
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


# UNet (latent space: 32x32 -> 16x16 -> 8x8 -> 4x4) 
class UNet(nn.Module):
    def __init__(self, in_ch=CHANNELS, base_ch=128):
        super().__init__()
        ch = base_ch  # 128

        self.stem = nn.Conv2d(in_ch, ch, 3, padding=1)  # 4 -> 128

        self.enc0_r1 = ResidualBlock(ch, ch)
        self.enc0_r2 = ResidualBlock(ch, ch)
        self.enc0_dn = Downsample(ch)  # -> 16x16

        self.enc1_r1 = ResidualBlock(ch, ch * 2)
        self.enc1_a1 = AttentionBlock(ch * 2)
        self.enc1_r2 = ResidualBlock(ch * 2, ch * 2)
        self.enc1_a2 = AttentionBlock(ch * 2)
        self.enc1_dn = Downsample(ch * 2)  # -> 8x8
        ch2 = ch * 2  # 256

        self.enc2_r1 = ResidualBlock(ch2, ch2 * 2)
        self.enc2_a1 = AttentionBlock(ch2 * 2)
        self.enc2_r2 = ResidualBlock(ch2 * 2, ch2 * 2)
        self.enc2_a2 = AttentionBlock(ch2 * 2)
        self.enc2_dn = Downsample(ch2 * 2)  # -> 4x4
        ch4 = ch2 * 2  # 512

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
        self.dec3_up = Upsample(ch4)  # -> 8x8

        self.dec2_r1 = ResidualBlock(ch4 + ch4, ch4)
        self.dec2_a1 = AttentionBlock(ch4)
        self.dec2_r2 = ResidualBlock(ch4 + ch4, ch2)
        self.dec2_a2 = AttentionBlock(ch2)
        self.dec2_r3 = ResidualBlock(ch2 + ch2, ch2)
        self.dec2_up = Upsample(ch2)  # -> 16x16

        self.dec1_r1 = ResidualBlock(ch2 + ch2, ch2)
        self.dec1_a1 = AttentionBlock(ch2)
        self.dec1_r2 = ResidualBlock(ch2 + ch2, ch)
        self.dec1_a2 = AttentionBlock(ch)
        self.dec1_r3 = ResidualBlock(ch + ch, ch)
        self.dec1_up = Upsample(ch)  # -> 32x32

        self.dec0_r1 = ResidualBlock(ch + ch, ch)
        self.dec0_r2 = ResidualBlock(ch + ch, ch)

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_ch, 3, padding=1),  # -> (B, 4, 32, 32)
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


#  Top-level Diffusion model 
class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(TIME_DIM, TIME_EMB_DIM)
        self.unet = UNet()

    def forward(self, x, t):
        temb = self.time_embedding(t)  # (B, TIME_EMB_DIM)
        return self.unet(x, temb)      # (B, 4, 32, 32)


# Decode & Sample 
@torch.no_grad()
def decode_latents(latents):
    """Unscale -> VAE decode -> [0,1] RGB (256x256)."""
    latents = latents.to(device) / SCALING_FACTOR
    with torch.amp.autocast("cuda", enabled=use_amp):
        imgs = vae.decode(latents).sample
    return (imgs / 2 + 0.5).clamp(0, 1)  # (B, 3, 256, 256)


@torch.no_grad()
def ddpm_sample(model, n=16, return_images=True):
    """Ancestral DDPM sampling in latent space. By default, returns decoded RGB images."""
    model.eval()
    x = torch.randn(n, CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    for t_val in reversed(range(T)):
        t_batch = torch.full((n,), t_val, device=device, dtype=torch.long)
        with torch.amp.autocast("cuda", enabled=use_amp):
            eps = model(x, t_batch)
        coef = (1.0 - alphas[t_val]) / sqrt_one_minus_ab[t_val]
        mean = sqrt_recip_a[t_val] * (x - coef * eps)
        x = mean + (torch.sqrt(betas_tilde[t_val]) * torch.randn_like(x) if t_val > 0 else 0)

    if return_images:
        return decode_latents(x)  # (n, 3, 256, 256) in [0,1]

    return x  # (n, 4, 32, 32) -- raw latents, do NOT clamp


#  FID 
def build_fid_metric(jpg_root):
    ds = JpgDataset(jpg_root)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    print("Pre-computing real FID stats...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="FID real"):
            metric.update(batch.to(device), real=True)
    return (
        metric,
        metric.real_features_sum.clone(),
        metric.real_features_cov_sum.clone(),
        metric.real_features_num_samples.clone(),
    )


def _restore_real(metric, mu, cov, n):
    metric.real_features_sum = mu.clone()
    metric.real_features_cov_sum = cov.clone()
    metric.real_features_num_samples = n.clone()


def compute_fid(model, metric, mu_r, cov_r, n_r, n_fake=128):
    metric.reset()
    _restore_real(metric, mu_r, cov_r, n_r)
    remaining = n_fake
    with torch.no_grad():
        while remaining > 0:
            n_batch = min(16, remaining)
            imgs = ddpm_sample(model, n=n_batch, return_images=True)
            metric.update(imgs, real=False)
            remaining -= n_batch
    model.train()
    return metric.compute().item()


#  Training 
def train():
    loader = make_latent_loader(DATA_ROOT_PT, batch_size=args.batch_size)

    model = Diffusion().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total:,} total | {trainable:,} trainable")

    fid_metric, mu_r, cov_r, n_r = build_fid_metric(DATA_ROOT_JPG)

    global_step = 0
    iter_losses = []
    fid_scores = []
    fid_steps = []
    epoch_losses = []
    start_epoch = 1

    #  Resume from checkpoint if provided 
    if args.resume_ckpt is not None:
        print(f"Resuming from checkpoint: {args.resume_ckpt}")
        ckpt = torch.load(args.resume_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        global_step = ckpt.get("global_step", 0)
        fid_scores = ckpt.get("fid_scores", [])
        fid_steps = ckpt.get("fid_steps", [])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed at epoch {start_epoch}, global_step {global_step}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        batch_losses = []
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for x0 in pbar:
            x0 = x0.to(device, non_blocking=True)  # (B, 4, 32, 32)
            t = torch.randint(0, T, (x0.shape[0],), device=device)
            xt, noise = q_sample(x0, t)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(xt, t)
                loss = F.mse_loss(pred, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            batch_losses.append(loss_val)
            iter_losses.append(loss_val)
            global_step += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}", step=global_step)

            # Logging 
            if global_step % args.log_loss_every == 0:
                plt.figure(figsize=(9, 4))
                plt.plot(iter_losses, linewidth=0.8)
                plt.xlabel("Iteration"); plt.ylabel("MSE Loss")
                plt.title(f"Training loss - step {global_step}")
                plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig("loss_curve.png", dpi=100); plt.close()

            if global_step % args.save_img_every == 0:
                imgs = ddpm_sample(model, n=16, return_images=True)  # (16, 3, 256, 256)
                vutils.save_image(imgs, f"{SAMPLE_DIR}/iter_{global_step:07d}.png", nrow=4)
                model.train()

            if global_step % args.calc_fid_every == 0:
                print(f"\nComputing FID at step {global_step}...")
                fid_val = compute_fid(model, fid_metric, mu_r, cov_r, n_r, n_fake=128)
                fid_scores.append(fid_val)
                fid_steps.append(global_step)
                print(f"[step {global_step}] FID: {fid_val:.2f}")

                plt.figure(figsize=(9, 4))
                plt.plot(fid_steps, fid_scores, marker="o", linewidth=1.2)
                plt.xlabel("Iteration"); plt.ylabel("FID")
                plt.title("FID over training")
                plt.grid(True, alpha=0.3); plt.tight_layout()
                plt.savefig("fid_curve.png", dpi=100); plt.close()
                model.train()

        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_losses.append(epoch_loss)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | loss {epoch_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.save_ckpt_every_epoch == 0:
            path = f"{CKPT_DIR}/ckpt_epoch_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "loss": epoch_loss,
                "fid_scores": fid_scores,
                "fid_steps": fid_steps,
            }, path)
            print(f"Saved checkpoint: {path}")

    plt.figure(figsize=(9, 4))
    plt.plot(range(1, args.epochs + 1), epoch_losses, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Avg Loss")
    plt.title("Epoch loss"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("epoch_loss_curve.png", dpi=100); plt.close()
    print("Training complete.")


if __name__ == "__main__":
    train()
