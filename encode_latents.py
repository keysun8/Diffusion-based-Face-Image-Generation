import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

#  Config 
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VAE_MODEL_ID = "CompVis/stable-diffusion-v1-4"
LATENT_SCALE = 0.18215  # SD's magic scaling constant (match at decode time)
LOADER_BATCH_SIZE = 256  # how many images the DataLoader fetches per iteration


#  Dataset 
class ImageFolderFlat(Dataset):
    """
    Walks img_dir recursively and returns (tensor, relative_path) pairs.
    Skips files whose corresponding .pt already exists in out_dir.
    """
    def __init__(self, img_dir: Path, out_dir: Path, transform):
        self.img_dir = img_dir
        self.out_dir = out_dir
        self.transform = transform

        all_paths = [
            p for p in img_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTS
        ]

        # filter already-encoded
        self.paths = []
        for p in all_paths:
            rel = p.relative_to(img_dir)
            out_p = out_dir / rel.with_suffix(".pt")
            if not out_p.exists():
                self.paths.append(p)

        skipped = len(all_paths) - len(self.paths)
        print(f"  Found  : {len(all_paths):,} images")
        print(f"  Skipped: {skipped:,} (already encoded)")
        print(f"  To do  : {len(self.paths):,}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)  # (3, 256, 256) in [-1, 1]
        except Exception as e:
            print(f"\n[WARN] Skipping corrupt file: {path}  ({e})")
            # return a black image so the batch doesn't break
            tensor = torch.zeros(3, 256, 256)

        rel = str(path.relative_to(self.img_dir))
        return tensor, rel


#  Args 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=Path, required=True,
                   help="Root folder containing images (searched recursively)")
    p.add_argument("--out_dir", type=Path, required=True,
                   help="Output folder for .pt latent files (mirrors subfolder structure)")
    p.add_argument("--batch_size", type=int, default=16,
                   help="Images per VAE forward pass — tune to fit VRAM")
    p.add_argument("--num_workers", type=int, default=8,
                   help="DataLoader workers")
    p.add_argument("--model_id", type=str, default=VAE_MODEL_ID,
                   help="HuggingFace model id or local path for the VAE")
    p.add_argument("--no_fp16", action="store_true",
                   help="Save latents as float32 (doubles disk usage)")
    p.add_argument("--sample", action="store_true",
                   help="Store sampled latent z instead of mean mu (adds randomness)")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (device == "cuda") else torch.float32
    print(f"\nDevice : {device}")
    print(f"VAE    : {args.model_id}\n")

    # Load VAE 
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=dtype,
    ).to(device)
    vae.eval()

    # Transform: resize+crop -> [-1, 1] 
    # Images are already 256x256, but this handles edge cases gracefully.
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(256),
        transforms.ToTensor(),                    # [0, 1]
        transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1]
    ])

    #  Dataset / DataLoader 
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolderFlat(args.img_dir, args.out_dir, transform)
    if len(dataset) == 0:
        print("Nothing to encode. All latents already exist.")
        return

    loader = DataLoader(
        dataset,
        batch_size=LOADER_BATCH_SIZE,  # data-loading batch, independent of VAE batch_size
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    #  Encode loop 
    save_dtype = torch.float16 if not args.no_fp16 else torch.float32
    total_saved = 0

    with tqdm(total=len(dataset), desc="Encoding", unit="img", dynamic_ncols=True) as pbar:
        for batch_imgs, batch_rels in loader:
            # Split into sub-batches of args.batch_size to control VRAM usage
            for start in range(0, len(batch_imgs), args.batch_size):
                imgs = batch_imgs[start: start + args.batch_size].to(device, dtype=dtype)
                rels = batch_rels[start: start + args.batch_size]

                # Encode -> DiagonalGaussian distribution
                dist = vae.encode(imgs).latent_dist

                if args.sample:
                    latents = dist.sample()  # mu + eps * sigma
                else:
                    latents = dist.mean       # deterministic mu <- default

                latents = latents * LATENT_SCALE  # scale to unit-ish variance

                # Save each latent individually
                for latent, rel in zip(latents, rels):
                    out_path = args.out_dir / Path(rel).with_suffix(".pt")
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(latent.cpu().to(save_dtype), out_path)
                    total_saved += 1

                pbar.update(len(rels))

    print(f"\nDone. Saved {total_saved:,} latents to {args.out_dir}")
    print(f"  Latent shape : (4, 32, 32)")
    print(f"  Dtype        : {'float16' if not args.no_fp16 else 'float32'}")
    print(f"  Est. size    : {total_saved * 4 * 32 * 32 * 2 / 1e9:.2f} GB (fp16)")


if __name__ == "__main__":
    main()
