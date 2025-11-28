"""
VAE Inference Helper (vae_with_pred_head edition)

Features
--------
1) Load a trained VAE checkpoint (from `vae_with_pred_head.py`).
2) Encode a single image -> latent (z, mu, logvar), reconstruct it, and compute errors.
3) Predict 7 targets from the latent using the model's prediction head, and (optionally) print
   them alongside ground-truth loaded from --target_json by filename index mapping.
4) Decode from a provided latent tensor to an image.
5) Systematically vary one latent *element* and visualize how the decoded image changes,
   while also recording how the 7 predicted targets change across the sweep.
6) Batch mode: predict targets for all images in a folder and compute dataset MAE.

Targets / Names
---------------
Order used throughout: ["E_max", "E_vertex", "Cdl", "HER_i0", "HER_alpha", "OER_i0", "OER_alpha"].

Scaling convention:
- During training, HER_i0 (idx 3) and OER_i0 (idx 5) are typically in log10 space.
- This script prints predictions in *linear* space (10** on those two entries).
- If your ground-truth in --target_json stores HER_i0 / OER_i0 in log10 space, pass --gt_i0_is_log10
  so they are exponentiated to match the printed prediction scale.

Ground-truth mapping via --target_json
--------------------------------------
- Provide a JSON/TXT file with one target vector per image in order.
- The image filename’s last run of digits selects the index: e.g. `0000.png` -> index 0,
  `0007.png` or `img_0007.jpg` -> index 7.
- Accepted JSON shapes:
  * Top-level list of lists: [[...7...], [...7...], ...]
  * Dict with key "targets" or "y" -> list of 7-D vectors
  * List of dicts keyed by the names above; they will be re-ordered into the standard order.
- TXT fallback: each line has 7 floats (comma or whitespace separated).

Notes
-----
- Expects the same preprocessing used during training:
  * Resize to 128, center crop to 128x128
  * Normalize to [-1, 1] with mean=std=[0.5, 0.5, 0.5]
- The latent is a spatial tensor of shape [1, 1, H, W] (e.g., H=W=8 -> 64 elements).
- You can choose a sweep element by flattened index (--sweep_index) or by row,col (--coords "r c").

Outputs
-------
When --image is used:
- infer_out/reconstruction.png        : Reconstructed image
- infer_out/orig_recon_grid.png       : Original (top) vs reconstruction (bottom)
- infer_out/latent.pt / latent.npy    : Encoded latent tensor
- infer_out/predictions.json          : Dict with predicted targets (linear & log entries) and, if provided,
                                        ground-truth (raw & aligned) + absolute errors and MAE/MSE summary

When sweeping a latent element:
- infer_out/sweep_r{r}_c{c}_S{steps}.png     : Grid of decoded images across the sweep values
- infer_out/sweep_values_r{r}_c{c}.txt       : The sweep values used
- infer_out/sweep_predictions_r{r}_c{c}.json : Per-step predicted targets (both log and linear)

Batch mode (--image_dir):
- infer_out/batch_results.jsonl : one JSON object per image with predictions (+ optional GT & abs error)
- infer_out/batch_metrics.json  : aggregate MAE per-dimension and overall

Quick examples
--------------
# 1) Encode & reconstruct (+ predict targets) and print/compare with GT
python vae_infer.py \
  --checkpoint path/to/vae_with_pred_head_state.pt \
  --image data/0007.png \
  --target_json data/targets.json \
  --gt_i0_is_log10 \
  --out_dir infer_out

# 2) Decode from a saved latent tensor
python vae_infer.py \
  --checkpoint path/to/vae_with_pred_head_state.pt \
  --latent infer_out/latent.pt \
  --out_dir infer_out

# 3) Sweep one latent element over a fixed range (by flattened index)
python vae_infer.py \
  --checkpoint path/to/vae_with_pred_head_state.pt \
  --image data/0007.png \
  --sweep_index 17 \
  --sweep_min -3.0 --sweep_max 3.0 --sweep_steps 11 \
  --out_dir infer_out

# 4) Sweep one latent element around mu ± K·sigma (requires --image)
python vae_infer.py \
  --checkpoint path/to/vae_with_pred_head_state.pt \
  --image data/0007.png \
  --sweep_index 17 \
  --use_mu_sigma --k_sigma 3.0 --sweep_steps 11 \
  --out_dir infer_out

# 5) Batch-predict an entire folder and compute MAE against target.txt
python vae_infer.py \
  --checkpoint outputs/vae_epoch_015.pt \
  --image_dir imgs \
  --target_json prediction_target/target.txt \
  --gt_i0_is_log10 \
  --out_dir infer_out
"""

import argparse
import math
import os
import json
import re
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms, utils as vutils

# Import the VAE class from the user's training script
try:
    from .vae_with_pred_head import VAE  # type: ignore
except Exception as e:
    from vae_with_pred_head import VAE
    # raise RuntimeError(
    #     "Could not import VAE from vae_with_pred_head.py. "
    #     "Place vae_infer.py next to vae_with_pred_head.py, or ensure it's on PYTHONPATH."
    # ) from e

# -------------------------------
# Utils
# -------------------------------

# Names for the 7 targets predicted by pred_head
#PRED_NAMES = ["E_max", "E_vertex", "Cdl", "HER_i0", "HER_alpha", "OER_i0", "OER_alpha"]
PRED_NAMES = ["HER_overpotential", "HER_i0"]

def mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.mse_loss(x, y).detach().cpu().item())


def psnr_from_mse(mse_val: float, data_range: float = 1.0) -> float:
    if mse_val <= 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse_val)


def denorm_to_01(x: torch.Tensor) -> torch.Tensor:
    """Convert from [-1,1] to [0,1] for saving/metrics."""
    return (x.clamp(-1, 1) + 1.0) * 0.5


def save_image01(x01: torch.Tensor, path: str) -> None:
    """Save a single image in [0,1] or a batch [B,3,H,W] in [0,1]."""
    vutils.save_image(x01, path)


def preprocess_image(path: str, device: torch.device) -> torch.Tensor:
    """Load and preprocess one image to [-1,1], shape [1,3,128,128]."""
    tfm = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1,1]
    ])
    img = Image.open(path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)  # [1,3,128,128]
    return x


def inverse_transform_preds(pred_log: torch.Tensor) -> torch.Tensor:
    """
    In training, log10 was applied to HER_i0 (idx 3) and OER_i0 (idx 5).
    This returns a copy with those dims exponentiated back to linear.
    """
    out = pred_log.clone()
    out[..., 1] = 10.0 ** out[..., 1]
    # out[..., 3] = 10.0 ** out[..., 3]
    # out[..., 5] = 10.0 ** out[..., 5]
    return out


def compute_metrics(x: torch.Tensor, recon: torch.Tensor) -> dict:
    # x and recon are both in [-1, 1]
    m = mse(x, recon)
    # Recompute MSE in [0,1] for PSNR (more standard)
    m01 = mse(denorm_to_01(x), denorm_to_01(recon))
    psnr = psnr_from_mse(m01, data_range=1.0)
    return {
        "mse": m,
        "mse_01": m01,
        "psnr": psnr,
    }


def load_model(checkpoint: str, device: torch.device) -> VAE:
    model = VAE().to(device)
    state = torch.load(checkpoint, map_location=device)
    # Some PyTorch versions need lazy modules initialized before loading.
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        with torch.no_grad():
            x_dummy = torch.zeros(1, 3, 128, 128, device=device)
            z_dummy, _, _ = model.encoder(x_dummy)
            _ = model.pred_head(z_dummy)
            _ = model.decoder(z_dummy)
        model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_latent(path: str, device: torch.device) -> torch.Tensor:
    if path.endswith(".pt"):
        z = torch.load(path, map_location=device)
    elif path.endswith(".npy"):
        z = torch.from_numpy(np.load(path)).to(device)
    else:
        raise ValueError("Unknown latent extension; expected .pt or .npy")
    if not isinstance(z, torch.Tensor):
        z = torch.as_tensor(z)
    return z.to(device)


def rc_to_flat(r: int, c: int, H: int, W: int) -> int:
    return r * W + c


def flat_to_rc(idx: int, H: int, W: int) -> Tuple[int, int]:
    if idx < 0 or idx >= H * W:
        raise ValueError(f"Flat index {idx} out of bounds for {H}x{W}")
    r = idx // W
    c = idx % W
    return r, c


def parse_image_index(image_path: str) -> int:
    """
    Extract a zero-based index from the image filename by taking the last
    contiguous run of digits. E.g. '.../0007.png' -> 7, 'img_0012.jpg' -> 12.
    """
    stem = os.path.splitext(os.path.basename(image_path))[0]
    m = list(re.finditer(r"(\d+)", stem))
    if not m:
        raise ValueError(f"Could not infer image index from filename: {image_path}")
    return int(m[-1].group(1))

def load_targets_list(json_or_txt_path: str):
    try:
        with open(json_or_txt_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("targets", data.get("y", data))

        # list of dicts
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return [[row["HER_overpotential"], row["HER_i0"]] for row in data]

        # list of lists
        if isinstance(data, list) and data and isinstance(data[0], list):
            L = len(data[0])
            if L == 2:
                return data
            if L >= 5:
                # 9-entry file: HER_overpotential is 4th, HER_i0 is 5th (1-based)
                # => indices 3 and 4 (0-based)
                return [[row[3], row[4]] for row in data]
            raise ValueError(f"Unsupported target length {L}; expected 2 or >=5.")
        return data
    except Exception:
        pass

    # TXT fallback
    rows = []
    with open(json_or_txt_path, "r") as f:
        for line in f:
            parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
            if not parts:
                continue
            if len(parts) == 2:
                rows.append([float(parts[0]), float(parts[1])])
            elif len(parts) >= 5:
                vals = [float(x) for x in parts]
                rows.append([vals[3], vals[4]])  # indices 3,4
            else:
                raise ValueError(f"Expected 2 values or >=5 per line, got {len(parts)}: '{line.strip()}'")
    return rows

# def load_targets_list(json_or_txt_path: str):
#     """
#     Load ground-truth targets from either:
#       1) JSON: top-level list, or dict with 'targets'/'y', or list of dicts keyed by PRED_NAMES
#       2) TXT: each line has 7 comma- or whitespace-separated floats
#     Returns a Python list of lists (N x 7).
#     """
#     # Try JSON first
#     try:
#         with open(json_or_txt_path, "r") as f:
#             data = json.load(f)
#         if isinstance(data, dict):
#             if "targets" in data:
#                 data = data["targets"]
#             elif "y" in data:
#                 data = data["y"]
#             else:
#                 raise ValueError(f"JSON dict must contain 'targets' or 'y'. Keys: {list(data.keys())}")
#         if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
#             data = [[row[name] for name in PRED_NAMES] for row in data]
#         if not isinstance(data, list):
#             raise ValueError("Targets JSON must be a list or a dict pointing to a list.")
#         if len(data) > 0 and len(data[0]) != 7:
#             raise ValueError(f"Each target vector must have length 7; got {len(data[0])}")
#         return data
#     except Exception:
#         pass  # fall back to TXT
#     # TXT fallback
#     rows = []
#     with open(json_or_txt_path, "r") as f:
#         for line in f:
#             parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
#             if not parts:
#                 continue
#             if len(parts) != 7:
#                 raise ValueError(f"Expected 7 values per line in TXT targets, got {len(parts)} on: '{line.strip()}'")
#             rows.append([float(x) for x in parts])
#     return rows


def sweep_one_latent(
    model: VAE,
    base_z: torch.Tensor,
    coord: Tuple[int, int],
    values: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    base_z: [1, 1, H, W]
    values: [S] values to assign to the (r,c) element
    Returns:
      decoded batch: [S, 3, 128, 128] in [-1, 1]
      preds_log:     [S, 2] predictions from pred_head (HER_i0 is log10 space)
    """
    assert base_z.ndim == 4 and base_z.shape[0] == 1 and base_z.shape[1] == 1, f"Expected [1,1,H,W], got {tuple(base_z.shape)}"
    r, c = coord
    outs = []
    preds = []
    with torch.no_grad():
        for v in values:
            z = base_z.clone()
            z[0, 0, r, c] = v
            x = model.decoder(z)
            y = model.pred_head(z)  # [1, 7]
            outs.append(x)
            preds.append(y)
    return torch.cat(outs, dim=0), torch.cat(preds, dim=0)  # [S,3,128,128], [S,7]


# -------------------------------
# Main
# -------------------------------

def infer_VAE(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model state_dict .pt")
    p.add_argument("--image", type=str, help="Path to a single input image to encode & reconstruct")
    p.add_argument("--latent", type=str, help="Path to a latent .pt/.npy to decode and optionally sweep around")
    p.add_argument("--out_dir", type=str, default="outputs", help="Output directory")

    # Batch prediction over a folder
    p.add_argument("--image_dir", type=str, default=None,
                   help="Folder containing images to batch-predict targets for (non-recursive).")
    p.add_argument("--extensions", type=str, default="png,jpg,jpeg,bmp",
                   help="Comma-separated list of allowed image extensions for --image_dir.")

    # Ground truth handling
    p.add_argument("--target_json", type=str, default=None,
                   help="Path to JSON (or .txt with 7 floats per line) containing ground-truth targets. "
                        "Index is matched to the image filename number (e.g., 0007.png -> index 7).")
    p.add_argument("--gt_i0_is_log10", action="store_true",
                   help="If set, assumes GT HER_i0 and OER_i0 (idx 3 and 5) are in log10; "
                        "they will be exponentiated to align with printed predictions.")

    # Sweep config
    p.add_argument("--sweep", action="store_true", help="If set, sweep chosen latent coords")
    p.add_argument("--coords", type=int, nargs="*", default=None, help="List of r c pairs, e.g. --coords 5 12 6 7")
    p.add_argument("--sweep_steps", type=int, default=11, help="Steps per sweep")
    p.add_argument("--sweep_min", type=float, default=-3.0, help="Min value for sweep (if not using mu/sigma)")
    p.add_argument("--sweep_max", type=float, default=3.0, help="Max value for sweep (if not using mu/sigma)")
    p.add_argument("--use_mu_sigma", action="store_true", help="Use mu±k*sigma for sweep bounds")
    p.add_argument("--k_sigma", type=float, default=3.0, help="k for mu±k*sigma sweep bounds")
    p.add_argument("--sweep_index", type=int, default=None, help="Flattened index (0..H*W-1) to sweep")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args(argv)
    device = torch.device(args.device)

    model = load_model(args.checkpoint, device)
    os.makedirs(args.out_dir, exist_ok=True)

    # =========================
    # Batch mode over a folder
    # =========================
    if args.image_dir:
        # Gather files
        exts = tuple("." + e.strip().lower() for e in args.extensions.split(",") if e.strip())
        all_names = sorted([n for n in os.listdir(args.image_dir)
                            if os.path.isfile(os.path.join(args.image_dir, n))
                            and n.lower().endswith(exts)])
        if not all_names:
            print(f"[batch] No images found in {args.image_dir} with extensions {exts}.")
            return

        # Load GT if provided
        tgt_list = None
        if args.target_json:
            tgt_list = load_targets_list(args.target_json)
            print(f"[batch] Loaded ground-truth with {len(tgt_list)} rows from: {args.target_json}")
        else:
            print("[batch] No --target_json provided; will skip MAE computation.")

        results_path = os.path.join(args.out_dir, "batch_results.jsonl")
        metrics_path = os.path.join(args.out_dir, "batch_metrics.json")
        n_with_gt = 0
        # sum_abs_err = np.zeros(7, dtype=np.float64)  
        sum_abs_err = np.zeros(2, dtype=np.float64)

        with open(results_path, "w") as fout:
            for name in all_names:
                fpath = os.path.join(args.image_dir, name)
                try:
                    x = preprocess_image(fpath, device)  # [-1,1]
                    with torch.no_grad():
                        z, mu, logvar = model.encoder(x)
                        pred_log = model.pred_head(z)      # [1,2] (log10 for i0)
                        pred = inverse_transform_preds(pred_log).squeeze(0).cpu().numpy()  # (2,)
                        # pred = inverse_transform_preds(pred_log).squeeze(0).detach().cpu().numpy()  # (7,)

                    idx = parse_image_index(fpath)
                    rec = {
                        "file": name,
                        "index_from_filename": idx,
                        "pred": pred.tolist(),
                        "pred_log": pred_log.squeeze(0).detach().cpu().tolist()
                    }

                    # If GT available and index valid, compute abs error
                    if tgt_list is not None and 0 <= idx < len(tgt_list):
                        gt_raw = np.asarray(tgt_list[idx], dtype=np.float64)
                        gt_aligned = gt_raw.copy()
                        # if args.gt_i0_is_log10:
                        #     gt_aligned[[3, 5]] = np.power(10.0, gt_aligned[[3, 5]])
                        if args.gt_i0_is_log10:
                            gt_aligned[1] = np.power(10.0, gt_aligned[1])   ## changed
                        abs_err = np.abs(pred - gt_aligned)
                        rec["gt_raw"] = gt_raw.tolist()
                        rec["gt"] = gt_aligned.tolist()
                        rec["abs_error"] = abs_err.tolist()
                        sum_abs_err += abs_err
                        n_with_gt += 1
                    elif tgt_list is not None:
                        rec["warn"] = f"Index {idx} out of range for targets of length {len(tgt_list)}"

                    fout.write(json.dumps(rec) + "\n")
                except Exception as e:
                    fout.write(json.dumps({"file": name, "error": str(e)}) + "\n")
                    print(f"[batch] Error processing {name}: {e}")

        # Aggregate MAE metrics
        metrics = {
            "names": PRED_NAMES,
            "count_images": len(all_names),
            "count_with_gt": n_with_gt,
        }
        if n_with_gt > 0:
            mae_per_dim = (sum_abs_err / float(n_with_gt)).tolist()
            metrics["MAE_per_dim"] = mae_per_dim
            metrics["MAE_overall"] = float(np.mean(sum_abs_err / float(n_with_gt)))
        else:
            metrics["MAE_per_dim"] = None
            metrics["MAE_overall"] = None

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[batch] Wrote per-file results to: {results_path}")
        print(f"[batch] Wrote aggregate metrics to: {metrics_path}")
        return

    # -------------------
    # Single-sample modes
    # -------------------


    # Case 1: decode from latent
    if args.latent:
        z = load_latent(args.latent, device)
        with torch.no_grad():
            recon = model.decoder(z)
        out_recon = os.path.join(args.out_dir, "decoded_from_latent.png")
        save_image01(denorm_to_01(recon), out_recon)
        print(f"[latent] Saved decoded image to: {out_recon}")

    # Case 2: encode & reconstruct from image
    if args.image:
        x = preprocess_image(args.image, device)                  # [-1,1]
        with torch.no_grad():
            # Access encoder directly to expose z, mu, logvar
            z, mu, logvar = model.encoder(x)
            recon = model.decoder(z)
            pred_log = model.pred_head(z)         # [1, 7] (HER_i0/OER_i0 in log10)
            pred = inverse_transform_preds(pred_log)
            # Return prediction head output (linear scale) when called programmatically
            return pred.squeeze(0).detach().cpu()
        metrics = compute_metrics(x, recon)

        # === Ground-truth lookup (optional) ===
        gt_raw = None   # as stored
        gt_aligned = None  # aligned to prediction scale (HER_i0/OER_i0 in linear)
        idx_used = None
        if args.target_json:
            idx_used = parse_image_index(args.image)
            tgt_list = load_targets_list(args.target_json)
            if 0 <= idx_used < len(tgt_list):
                gt_raw = np.asarray(tgt_list[idx_used], dtype=np.float64)  # shape (7,)
                gt_aligned = gt_raw.copy()
                if args.gt_i0_is_log10:
                    gt_aligned[1] = 10.0 ** gt_aligned[1]
                    # gt_aligned[[3, 5]] = np.power(10.0, gt_aligned[[3, 5]])
            else:
                print(f"[warn] Image index {idx_used} out of range for targets (len={len(tgt_list)}); GT omitted.")

        # Save outputs
        out_latent_pt = os.path.join(args.out_dir, "latent.pt")
        out_latent_npy = os.path.join(args.out_dir, "latent.npy")
        out_recon = os.path.join(args.out_dir, "reconstruction.png")
        out_side_by_side = os.path.join(args.out_dir, "orig_recon_grid.png")
        out_preds = os.path.join(args.out_dir, "predictions.json")

        with open(out_preds, "w") as f:
            pred_lin = pred.squeeze(0).detach().cpu().tolist()
            payload = {
                "names": PRED_NAMES,
                "index_from_filename": idx_used,
                "pred_log": pred_log.squeeze(0).detach().cpu().tolist(),
                "pred": pred_lin,  # HER_i0/OER_i0 already brought to linear
                "recon_metrics": metrics,
            }
            if gt_raw is not None:
                payload["gt_raw"] = gt_raw.tolist()
                payload["gt"] = gt_aligned.tolist() if gt_aligned is not None else None
                # errors computed in the aligned (linear) scale
                abs_err = (np.abs(np.array(pred_lin, dtype=np.float64) - gt_aligned)).tolist()
                payload["abs_error"] = abs_err
                payload["MAE"] = float(np.mean(abs_err))
                payload["MSE"] = float(np.mean((np.array(pred_lin) - gt_aligned) ** 2))
            json.dump(payload, f, indent=2)
        print("[predict] Targets saved to:", out_preds)

        # Console pretty print: predictions (linear) and, if available, ground-truth + errors
        pretty_pred = {k: v for k, v in zip(PRED_NAMES, pred_lin)}
        print("[predict] Predicted (HER_i0/OER_i0 in linear scale):")
        print(json.dumps(prety_pred , indent=2))
        if gt_aligned is not None:
            pretty_gt = {k: float(v) for k, v in zip(PRED_NAMES, gt_aligned.tolist())}
            abs_err = {k: abs(pretty_pred[k] - pretty_gt[k]) for k in PRED_NAMES}
            print("[predict] Ground-truth (aligned to prediction scale):")
            print(json.dumps(pretty_gt, indent=2))
            print("[predict] Absolute error:")
            print(json.dumps(abs_err, indent=2))

        torch.save(z.detach().cpu(), out_latent_pt)
        np.save(out_latent_npy, z.detach().cpu().numpy())
        save_image01(denorm_to_01(recon), out_recon)
        # Side-by-side grid (original on top row, recon on bottom row)
        grid = vutils.make_grid(torch.cat([denorm_to_01(x), denorm_to_01(recon)], dim=0), nrow=1)
        vutils.save_image(grid, out_side_by_side)
        print(f"[image] Saved recon: {out_recon}")
        print(f"[image] Saved orig/recon grid: {out_side_by_side}")
        print(f"[image] Metrics: {metrics}")

        # Optional sweep
        if args.sweep:
            # Determine coords to sweep
            if args.coords is not None and len(args.coords) >= 2 and len(args.coords) % 2 == 0:
                coords: List[Tuple[int, int]] = [(args.coords[i], args.coords[i + 1]) for i in range(0, len(args.coords), 2)]
            elif args.sweep_index is not None:
                H, W = z.shape[2], z.shape[3]
                coords = [flat_to_rc(args.sweep_index, H, W)]
            else:
                raise ValueError("Provide --coords r c (pairs) or --sweep_index to choose which latent element(s) to sweep.")

            for (r, c) in coords:
                if args.use_mu_sigma:
                    mu_rc = float(mu[0, 0, r, c].detach().cpu().item())
                    std_rc = float(torch.exp(0.5 * logvar[0, 0, r, c]).detach().cpu().item())
                    center, std = mu_rc, std_rc
                    vmin = center - args.k_sigma * std
                    vmax = center + args.k_sigma * std
                else:
                    vmin, vmax = args.sweep_min, args.sweep_max

                values = torch.linspace(vmin, vmax, steps=args.sweep_steps, device=device)
                decoded_batch, preds_log = sweep_one_latent(model, z, (r, c), values, device)  # [S,3,128,128], [S,7]
                preds = inverse_transform_preds(preds_log)

                # Build a tidy grid with labels
                grid = vutils.make_grid(denorm_to_01(decoded_batch), nrow=args.sweep_steps)
                out_grid = os.path.join(args.out_dir, f"sweep_r{r}_c{c}_S{args.sweep_steps}.png")
                vutils.save_image(grid, out_grid)

                # Save sweep predictions
                out_sweep_preds = os.path.join(args.out_dir, f"sweep_predictions_r{r}_c{c}.json")
                with open(out_sweep_preds, "w") as f:
                    json.dump({
                        "coord": [r, c],
                        "names": PRED_NAMES,
                        "values": [float(v) for v in values.detach().cpu().tolist()],
                        "preds_log": preds_log.detach().cpu().tolist(),
                        "preds": preds.detach().cpu().tolist()
                    }, f, indent=2)

                # Also save the values used for reference (txt)
                out_vals = os.path.join(args.out_dir, f"sweep_values_r{r}_c{c}.txt")
                with open(out_vals, "w") as f:
                    f.write("\n".join([f"{i:02d}: {float(v):.6f}" for i, v in enumerate(values.detach().cpu().tolist())]))

                print(f"[sweep] Saved grid: {out_grid}")
                print(f"[sweep] Values -> {out_vals}")
                print(f"[sweep] Predictions -> {out_sweep_preds}")

    if not args.image and not args.latent and not args.image_dir:
        print("Nothing to do. Provide --image to encode/reconstruct, --latent to decode, or --image_dir for batch mode.")
        return


if __name__ == "__main__":
    infer_VAE()
