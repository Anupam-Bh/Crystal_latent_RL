import os
import glob
import math
from typing import List, Tuple
from PIL import Image
import json,re  

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils as vutils
from torchsummary import summary

# -------------------------------
# Model blocks
# -------------------------------

#class SelfAttention(nn.Module):
#    def __init__(self, n_heads: int, embd_dim: int, in_proj_bias: bool = True, out_proj_bias: bool = True):
#        super().__init__()
#        assert embd_dim % n_heads == 0, "embd_dim must be divisible by n_heads"
#        self.n_heads = n_heads
#        self.d_heads = embd_dim // n_heads
#        self.in_proj = nn.Linear(embd_dim, 3 * embd_dim, bias=in_proj_bias)
#        self.out_proj = nn.Linear(embd_dim, embd_dim, bias=out_proj_bias)
#
#    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
#        # x: [B, N, C]
#        B, N, C = x.shape
#        q, k, v = self.in_proj(x).chunk(3, dim=-1)  # [B, N, C] each
#        # reshape to heads
#        def to_heads(t):
#            return t.view(B, N, self.n_heads, self.d_heads).transpose(1, 2)  # [B, H, N, Dh]
#        q, k, v = map(to_heads, (q, k, v))
#        attn = q @ k.transpose(-1, -2) / math.sqrt(self.d_heads)  # [B, H, N, N]
#        if causal_mask:
#            mask = torch.ones_like(attn, dtype=torch.bool).triu(1)
#            attn = attn.masked_fill(mask, float('-inf'))
#        attn = attn.softmax(dim=-1)
#        out = attn @ v  # [B, H, N, Dh]
#        out = out.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]
#        return self.out_proj(out)
#
#class AttentionBlock(nn.Module):
#    def __init__(self, channels: int):
#        super().__init__()
#        self.norm = nn.GroupNorm(32, channels)
#        self.attn = SelfAttention(1, channels)
#
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        residual = x
#        x = self.norm(x)
#        B, C, H, W = x.shape
#        x = x.view(B, C, H * W).transpose(1, 2)  # [B, HW, C]
#        x = self.attn(x)
#        x = x.transpose(1, 2).view(B, C, H, W)
#        return x + residual
#
#class ResidualBlock(nn.Module):
#    def __init__(self, in_channels: int, out_channels: int):
#        super().__init__()
#        self.norm1 = nn.GroupNorm(32, in_channels)
#        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
#        self.norm2 = nn.GroupNorm(32, out_channels)
#        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
#        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)
#
#    def forward(self, x: torch.Tensor) -> torch.Tensor:
#        residual = x
#        x = self.norm1(x)
#        x = F.silu(x)
#        x = self.conv1(x)
#        x = self.norm2(x)
#        x = F.silu(x)
#        x = self.conv2(x)
#        return x + self.skip(residual)

# -------------------------------
# Encoder / Decoder
# -------------------------------

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            # 64x64 -> 64x64  
            nn.Conv2d(3, 128, 3, padding=1),
            nn.SiLU(),
            #ResidualBlock(128, 128),
            #AttentionBlock(128),
            # 64 -> 32
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.SiLU(),
            #ResidualBlock(256, 256),
            #AttentionBlock(256),
            # 32 -> 16
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.SiLU(),
            #ResidualBlock(256, 256),
            #AttentionBlock(256),
            # 16 -> 8
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            #ResidualBlock(512, 512),
            #AttentionBlock(512),
            nn.SiLU(),
            #nn.Conv2d(512, 512, 3, stride=2, padding=1),
            #nn.SiLU(),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 2, 1),
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for m in self.blocks:
            x = m(x)
        mu, logvar = torch.chunk(x, 2, dim=1)  # 4 each
        logvar = torch.clamp(logvar, -30, 20)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        #z = z * 0.18215  # SD scaling
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.ModuleList([
            nn.Conv2d(1, 512, 3, padding=1),
            nn.SiLU(),
            #ResidualBlock(512, 512),
            #AttentionBlock(512),
            # 8 -> 16
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.SiLU(),
            #ResidualBlock(256, 256),
            #AttentionBlock(256),
            # 16 -> 32
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.SiLU(),
            #ResidualBlock(256, 256),
            #AttentionBlock(256),
            # 32 -> 64
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            #ResidualBlock(128, 128),
            #AttentionBlock(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, 3, padding=1),
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z / 0.18215
        x = z
        for m in self.net:
            x = m(x)
        return torch.tanh(x)  # outputs in [-1, 1]

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # --- Prediction head on latent ---
        # Takes flattened z (B, 1, H, W) -> many targets e.g. [E_max, E_vertex, Cdl, HER_i0, HER_alpha, OER_i0, OER_alpha]
        # Only use 2 targets   HER_overpotential  and   HER_i0 
        self.pred_head = nn.Sequential(
            nn.Flatten(),           # infer H*W at runtime
            nn.LazyLinear(128),
            nn.SiLU(),
            nn.Linear(128, 2),  # 2 targets
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encoder(x)
        recon = self.decoder(z)
        #return recon, z, mu, logvar
        pred = self.pred_head(z)
        return recon, z, mu, logvar, pred

# -------------------------------
# Dataset for a single folder of images (no labels)
# -------------------------------

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

class ImageFolderNoLabels(Dataset):
    def __init__(self, root: str, transform=None, sort: bool = True):
        self.root = root
        self.transform = transform
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(root, f"*{ext}")))
        # also pick up zero-padded like 001.png automatically via glob
        self.paths = sorted(files) if sort else files
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}. Supported: {IMG_EXTS}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, os.path.basename(path)  # return filename for saving

# -------------------------------
# Training utilities
# -------------------------------

def vae_loss(recon_x, x, mu, logvar, recon_weight=1.0, kld_weight=1e-4):
    # MSE reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    # KL divergence per element, averaged over batch
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_weight * recon_loss + kld_weight * kld, recon_loss.detach(), kld.detach()


def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    # from [-1,1] to [0,1]
    return (x + 1.0) / 2.0


def train_and_reconstruct(
    images_dir: str,
    out_dir: str = "outputs",
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 2e-4,
    num_workers: int = 2,
    val_split: float = 0.1,
    kld_weight: float = 1e-4,
    seed: int = 42,
    # --- NEW: optional supervised head on z ---
    targets_json: str = None,   # path to JSON list saved by save_prediction_head
    pred_weight: float = 1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    os.makedirs(out_dir, exist_ok=True)
    recon_dir = os.path.join(out_dir, "reconstructions")
    os.makedirs(recon_dir, exist_ok=True)

    ##Make 128*128 input
    transform = transforms.Compose([
        transforms.Resize(128, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),  # -> [-1, 1]
    ])

    dataset = ImageFolderNoLabels(images_dir, transform=transform, sort=True)

    # ---- Load targets (HER_i0 at idx 3, OER_i0 at idx 5). We'll map by filename index. ----
    targets_all = None     
    if targets_json is not None and os.path.exists(targets_json):  ## when targets_json is entered as input argument, then only this error is calculated
        with open(targets_json, "r") as f:
            targets_all = json.load(f)  # list[list]

        def name_to_idx(name: str):# -> int | None:
            # Extract trailing digits from filenames like 00012.png, img_00012.png, etc.
            m = re.findall(r"\d+", name)
            #print(m)
            return int(m[-1]) if m else None

        def get_batch_targets(names: List[str]) -> torch.Tensor:
            idxs = [name_to_idx(n) for n in names]
            #print(idxs)
            #ys = [[targets_all[i][3], targets_all[i][5]] for i in idxs]  # HER_i0, OER_i0
            #y = torch.tensor(ys, dtype=torch.float32, device=device)
            #return torch.log10(torch.clamp(y, min=1e-30))  # log10 transform as requested
            # Full 7-vector: [E_max, E_vertex, Cdl, HER_i0, HER_alpha, OER_i0, OER_alpha]
            # ys = [targets_all[i][:7] for i in idxs]
            ys = [[targets_all[i][3], targets_all[i][5]] for i in idxs]  # HER_overpot, HER_i0
            y = torch.tensor(ys, dtype=torch.float32, device=device)
            # Apply log10 only to HER_i0 (idx 3) and OER_i0 (idx 5)
            eps = 1e-30
            #y[:, 0] = torch.log10(torch.clamp(y[:, 0], min=eps))  
            # y[:, 3] = torch.log10(torch.clamp(y[:, 3], min=eps))
            y[:, 1] = torch.log10(torch.clamp(y[:, 1], min=eps))  # apply log10 to both columns
            return y            


    # Split into train/val
    n = len(dataset)
    n_val = max(1, int(val_split * n))
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    model = VAE().to(device)
    #summary(model, input_size=(3,128,128))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        running_pred = 0.0
        for imgs, _names in train_loader:
            #print(_names)
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
#            recon, z, mu, logvar = model(imgs)
#            loss, rec_l, kld_l = vae_loss(recon, imgs, mu, logvar, kld_weight=kld_weight)
#            loss.backward()
            recon, z, mu, logvar, pred = model(imgs)
            base_loss, rec_l, kld_l = vae_loss(recon, imgs, mu, logvar, kld_weight=kld_weight)

            # Optional prediction loss
            pred_l = torch.tensor(0.0, device=device)
            if targets_all is not None:
                y_log = get_batch_targets(_names)
                pred_l = F.mse_loss(pred, y_log, reduction="mean")

            loss = base_loss + pred_weight * pred_l
            loss.backward()
            optimizer.step()
            running += loss.item()
            running_pred += float(pred_l.item()) if torch.is_tensor(pred_l) else 0.0   ## added
            global_step += 1

        avg_train = running / max(1, len(train_loader))
        avg_train_pred = running_pred / max(1, len(train_loader))  ### added

        # quick val
        model.eval()
        with torch.no_grad():
            val_losses = []
            val_pred_losses = []
            for imgs, _names in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                #recon, z, mu, logvar = model(imgs)
                #loss, _, _ = vae_loss(recon, imgs, mu, logvar, kld_weight=kld_weight)
                recon, z, mu, logvar, pred = model(imgs)
                loss, _, _ = vae_loss(recon, imgs, mu, logvar, kld_weight=kld_weight)
                if targets_all is not None:
                    y_log = get_batch_targets(_names)
                    pred_l = F.mse_loss(pred, y_log, reduction="mean")
                    loss = loss + pred_weight * pred_l
                    val_pred_losses.append(pred_l.item())
                val_losses.append(loss.item())
            avg_val = sum(val_losses) / max(1, len(val_losses))
            avg_val_pred = (sum(val_pred_losses) / max(1, len(val_pred_losses))) if val_pred_losses else 0.0

        #print(f"Epoch {epoch}/{num_epochs} - train_loss: {avg_train:.4f}  val_loss: {avg_val:.4f}")
        if targets_all is not None:
            print(f"Epoch {epoch}/{num_epochs} - train_loss: {avg_train:.4f} (pred {avg_train_pred:.4f})  "
                  f"val_loss: {avg_val:.4f} (pred {avg_val_pred:.4f})")
        else:
            print(f"Epoch {epoch}/{num_epochs} - train_loss: {avg_train:.4f}  val_loss: {avg_val:.4f}")

        # Save a small grid preview
        model.eval()
        with torch.no_grad():
            sample_imgs, _ = next(iter(val_loader)) if len(val_loader) > 0 else next(iter(train_loader))
            sample_imgs = sample_imgs.to(device)
            sample_recon, *_ = model(sample_imgs)
            grid = vutils.make_grid(torch.cat([denorm_to_01(sample_imgs[:8]), denorm_to_01(sample_recon[:8])], dim=0), nrow=8)
            vutils.save_image(grid, os.path.join(out_dir, f"epoch_{epoch:003d}_grid.png"))

        # checkpoint
        torch.save(model.state_dict(), os.path.join(out_dir, f"vae_epoch_{epoch:003d}.pt"))

#    # Final reconstruction pass for all images; save with matching filenames
#    model.eval()
#    with torch.no_grad():
#        for imgs, names in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
#            imgs = imgs.to(device)
#            recon, *_ = model(imgs)
#            recon = denorm_to_01(recon).cpu()
#            for img_tensor, name in zip(recon, names):
#                vutils.save_image(img_tensor, os.path.join(recon_dir, name))

    print(f"Done. Reconstructions saved to: {recon_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train VAE on images in a single folder and reconstruct them.")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to folder containing images (e.g., 001.png, 002.png, ...)")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Where to write checkpoints and reconstructions")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--kld_weight", type=float, default=1e-4)
    parser.add_argument("--targets_json", type=str, default=None, help="Path to JSON list with targets")
    parser.add_argument("--pred_weight", type=float, default=1.0, help="Weight for prediction loss term")
    args = parser.parse_args()

    train_and_reconstruct(
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        num_workers=args.workers,
        kld_weight=args.kld_weight,
        targets_json=args.targets_json,
        pred_weight=args.pred_weight,
        )

