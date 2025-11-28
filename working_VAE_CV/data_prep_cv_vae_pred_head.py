
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy  #### to avoid mutation of p

import os
from PIL import Image
import  matplotlib.pyplot as plt

# ---- Import user's simulator and defaults ----
# NOTE: unchanged: we still call your simulator and immediately rasterize to an image
try:
    from .working_CV_model import simulate_cv_multi_with_watersplitting, p as DEFAULT_P, couples as DEFAULT_COUPLES
except ImportError:
    # allows running this file directly (python data_prep_cv_vae_pred_head.py)
    from working_CV_model import simulate_cv_multi_with_watersplitting, p as DEFAULT_P, couples as DEFAULT_COUPLES


# --------------------------
# Utilities
# --------------------------
def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _resample_to_fixed_n(t: np.ndarray, y: np.ndarray, n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly resample y(t) to n_points across [t.min(), t.max()].
    """
    t = np.asarray(t, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1e-12

    tt = np.linspace(t[0], t[-1], n_points, dtype=np.float64)
    yy = np.interp(tt, t, y)
    return tt, yy

@dataclass
class CoupleSamplerConfig:
    """Ranges used to randomize redox couples for a new 'solution' each episode."""
    max_couples: int = 2
    min_couples: int = 1
    E0_range: Tuple[float, float] = (-0.5, 0.9)     # V
    D_range: Tuple[float, float] = (3e-4, 3e-3)     # m^2/s (diffusivities)
    CObulk_range: Tuple[float, float] = (0.2, 1.0)  # mol/m^3 (arbitrary units OK)
    k0_range: Tuple[float, float] = (1e-6, 1e-3)    # m/s (when used)
    alpha_range: Tuple[float, float] = (0.3, 0.7)
    prob_quasi_rev: float = 0.6                     # prob. of using BV (quasi-reversible) boundary
    n_options: Tuple[int, ...] = (1, 2)             # electron numbers to sample

# --------------------------
# Image Dataset (grayscale CV plots)
# --------------------------
class CVImageDataset(Dataset):
    """
    Holds (N, H, W) float32 images in [0,1] and yields tensors shaped (1, H, W).
    """
    def __init__(self, images_hw: np.ndarray):
        super().__init__()
        assert images_hw.ndim == 3, "Expect (N, H, W) array of grayscale images"
        self.imgs = images_hw.astype(np.float32)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx: int):
        x = self.imgs[idx][None, ...]  # (1,H,W)
        return torch.from_numpy(x)


def build_dataloaders_from_arrays(
    train_imgs: np.ndarray,
    val_imgs: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    ds_tr = CVImageDataset(train_imgs)
    ds_va = CVImageDataset(val_imgs)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return dl_tr, dl_va


def load_numpy_datasets(in_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (train_images.npy, val_images.npy). Each shape is (N, H, W) with values in [0,1].
    """
    import os
    tr = np.load(os.path.join(in_dir, "train_images.npy"))
    va = np.load(os.path.join(in_dir, "val_images.npy"))
    return tr, va


# --------------------------
# Rasterization helpers
# --------------------------
def _hist2d_rasterize(
    E: np.ndarray,
    I: np.ndarray,
    Emin_clip: float = -1.2,
    Emax_clip: float = +1.2,
    Imin_clip: float = -30e-3,
    Imax_clip: float = +30e-3,
    H: int = 128,
    W: int = 128) -> np.ndarray:
    """
    Convert (E, I) curve to a **binary** grayscale image using a 2D histogram.
    The plot is cropped to the given bounds and returned as float32 image with values in {0,1}.
    Axis mapping: x <- E (columns), y <- I (rows, bottom=Imin, top=Imax).
    """
    E = np.asarray(E, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)

    # Clip to the requested window
    E = np.clip(E, Emin_clip, Emax_clip)
    I = np.clip(I, Imin_clip, Imax_clip)

    # 2D histogram: rows correspond to I bins, columns to E bins
    bins_E = np.linspace(Emin_clip, Emax_clip, W + 1)
    bins_I = np.linspace(Imin_clip, Imax_clip, H + 1)
    counts, _, _ = np.histogram2d(I, E, bins=[bins_I, bins_E])  # shape (H, W)
    #counts, _, _ = np.histogram2d(E, I)

    # ---- BINARY: mark presence (1) vs absence (0), no normalization, no blur ----
    #img = plt.imshow(counts>0,origin='lower', interpolation='none')
    #print(img)
    img = (counts > 0).astype(np.float32)
    #plt.plot(E,I,linewidth=5)
    #plt.xlim([Emin_clip, Emax_clip])
    #plt.ylim([Imin_clip, Imax_clip])
    #plt.show()
    return img

# def _hist2d_rasterize(
#     E: np.ndarray,
#     I: np.ndarray,
#     Emin_clip: float = -1.5,
#     Emax_clip: float = +1.5,
#     Imin_clip: float = -50e-3,
#     Imax_clip: float = +50e-3,
#     H: int = 128,
#     W: int = 128,
#     blur_iters: int = 1,
# ) -> np.ndarray:
#     """
#     Convert (E, I) curve to a grayscale image using a 2D histogram + light box-blur.
#     The plot is cropped to the given bounds and returned as float32 image in [0,1].
#     Axis mapping: x <- E (columns), y <- I (rows, bottom=Imin, top=Imax).
#     """
#     E = np.asarray(E, dtype=np.float64)
#     I = np.asarray(I, dtype=np.float64)

#     # Clip to the requested window
#     E = np.clip(E, Emin_clip, Emax_clip)
#     I = np.clip(I, Imin_clip, Imax_clip)

#     # 2D histogram: rows correspond to I bins, columns to E bins
#     bins_E = np.linspace(Emin_clip, Emax_clip, W + 1)
#     bins_I = np.linspace(Imin_clip, Imax_clip, H + 1)
#     counts, _, _ = np.histogram2d(I, E, bins=[bins_I, bins_E])  # shape (H, W)

#     # Normalize counts -> [0,1]
#     if counts.max() > 0:
#         img = counts / counts.max()
#     else:
#         img = counts

#     # Lightweight box blur (thickens the trace), repeated blur_iters times
#     if blur_iters > 0:
#         k = np.array([[1,1,1],
#                       [1,1,1],
#                       [1,1,1]], dtype=np.float64) / 9.0
#         for _ in range(blur_iters):
#             img = _convolve2d_validpad(img, k)

#     # Re-normalize to [0,1]
#     m = img.max()
#     if m > 0:
#         img = img / m
#     img = img.astype(np.float32)

#     return img


#def _convolve2d_validpad(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
#    """
#    2D convolution with replicate padding at the border (simple and dependency-free).
#    """
#    H, W = img.shape
#    kh, kw = kernel.shape
#    pad_h, pad_w = kh // 2, kw // 2
#    # replicate-pad
#    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
#    out = np.zeros_like(img, dtype=np.float64)
#    for i in range(H):
#        for j in range(W):
#            patch = padded[i:i+kh, j:j+kw]
#            out[i, j] = float(np.sum(patch * kernel))
#    return out

#--------------------------------
# Helper to identify E for given I
#-------------------------------
def all_E_where_I_equals(E, I, I_target):
    E = np.asarray(E)
    I = np.asarray(I)
    d = I - I_target

    # indices where the line segment crosses the target (sign change or exact hit)
    hit = (d == 0)
    cross = (d[:-1] * d[1:] < 0) | hit[:-1] | hit[1:]
    idx = np.where(cross)[0]

    E_solutions = []
    for k in idx:
        I1, I2 = I[k], I[k+1]
        E1, E2 = E[k], E[k+1]
        if I2 == I1:  # flat segment at target (rare)
            if I1 == I_target:
                E_solutions.extend([E1, E2])
        else:
            frac = (I_target - I1) / (I2 - I1)
            E_solutions.append(E1 + frac * (E2 - E1))

    # optional: dedupe near-identical values
    E_solutions = np.array(E_solutions, dtype=float)
    E_solutions.sort()
    return E_solutions


# --------------------------
# Simulation wrapper -> image
# --------------------------
@dataclass
class SimConfig:
    image_size: int = 128
    #blur_iters: int = 1
    vary_params: bool = True
    add_noise_std: float = 0.0
    seed: int = 123

def simulate_one_cv_image(
    cfg: SimConfig,
    base_p: Optional[Dict] = None,
    couples: Optional[List[Dict]] = None,
    rng: Optional[np.random.Generator] = None,
    index: int = 0,
    HER_i0: float|None=None,
    ) -> np.ndarray:
    """
    Run one CV simulation, immediately rasterize a 2D grayscale CV plot.
    Returns image with shape (H, W) in [0,1].
    """
    if index % 100 == 0:
        print(index)

    if rng is None:
        rng = np.random.default_rng()
    H = W = int(cfg.image_size)

    # Copy defaults to avoid mutation
    #p = dict(DEFAULT_P if base_p is None else base_p)
    #cps = [dict(c) for c in (DEFAULT_COUPLES if couples is None else couples)]
    p   = deepcopy(DEFAULT_P if base_p is None else base_p)
    cps = deepcopy(DEFAULT_COUPLES if couples is None else couples)

    # Simple randomization to diversify dataset
    if cfg.vary_params:
        p["scan_rate"] = float(rng.uniform(0.005, 0.05))    # 5–50 mV/s
        p['E_max'] = 1
        p['E_vertex'] = -0.3
        # p["E_max"]     = float(rng.uniform(0.3, 0.9))       # V
        # p["E_vertex"]  = float(rng.uniform(-0.9, -0.3))     # V
        p["Cdl"]       = float(rng.uniform(2e-4, 2e-2))     # F/m^2
        p["n_cycles"]  = int(rng.integers(2, 10))

        if "HER" in p:
            #print(rng.integers(0, 2))
            p["HER"]["enabled"] = bool(rng.integers(1, 2))
            if HER_i0:
                p["HER"]["i0"] = HER_i0
            else:
                p["HER"]["i0"] = float(10 ** rng.uniform(-2, 1))  ## 0.01 to 10

            p["HER"]["alpha"] =float(rng.uniform(0.45,0.55)) 
        if "OER" in p:
            p["OER"]["enabled"] = True
            p["OER"]["i0"] = 10 **-13    ### keeping  it near zero because we are only optimizing HER
            p["OER"]["alpha"] =0.12
            # p["OER"]["enabled"] = bool(rng.integers(1, 2))
            # p["OER"]["i0"] = float(10 ** rng.uniform(-15, -12))
            # p["OER"]["alpha"] =float(rng.uniform(0.11,0.13))


    #    for c in cps:
    #        c["DO"] *= float(rng.uniform(0.7, 1.3))
    #        c["DR"] *= float(rng.uniform(0.7, 1.3))
    #        if c.get("k0") is not None:
    #            c["k0"] *= float(rng.uniform(0.5, 2.0))
        ### Randomize the redox couples
        cp_cfg = CoupleSamplerConfig()
        n_cpl = int(rng.integers(cp_cfg.min_couples, cp_cfg.max_couples + 1))
        cps: List[Dict] = []
        for i in range(n_cpl):
            n = int(rng.choice(cp_cfg.n_options))
            E0 = float(rng.uniform(*cp_cfg.E0_range))
            DO = float(10 ** rng.uniform(np.log10(cp_cfg.D_range[0]), np.log10(cp_cfg.D_range[1])))
            DR = float(10 ** rng.uniform(np.log10(cp_cfg.D_range[0]), np.log10(cp_cfg.D_range[1])))
            CObulk = float(rng.uniform(*cp_cfg.CObulk_range))
            CRbulk = 0.0
            if rng.random() < cp_cfg.prob_quasi_rev:
                k0 = float(10 ** rng.uniform(np.log10(cp_cfg.k0_range[0]), np.log10(cp_cfg.k0_range[1])))
                alpha = float(rng.uniform(*cp_cfg.alpha_range))
            else:
                k0 = None
                alpha = 0.5
            cps.append(
                dict(
                    label=f"C{i}",
                    n=n,
                    E0=E0,
                    DO=DO,
                    DR=DR,
                    CObulk=CObulk,
                    CRbulk=CRbulk,
                    k0=k0,
                    alpha=alpha,
                )
             )

    # ----- Simulate -----
    # We take the last-cycle E(t), I(t)
    E_last, I_last, *_rest, x, t_last, CO, CR = simulate_cv_multi_with_watersplitting(p, cps)

    # Optional measurement noise on current
    I_last = np.array(I_last, dtype=np.float64)
    if cfg.add_noise_std > 0.0:
        I_last = I_last + rng.normal(0.0, cfg.add_noise_std, size=I_last.shape)
    
    #Emin_clip: float = -1.0
    #Emax_clip: float = +1.0
    #Imin_clip: float = -40e-3
    #Imax_clip: float = +40e-3
    #img = plt.plot(E_last,I_last,linewidth =4)
    #plt.xlim(Emin_clip, Emax_clip)
    #plt.ylim(Imin_clip,Imax_clip)
    
    # plt.show()
    # ----- Rasterize immediately after obtaining E and I -----
    #img = _hist2d_rasterize(
    #    E=np.asarray(E_last, dtype=np.float64),
    #    I=np.asarray(I_last, dtype=np.float64),
    #    Emin_clip=-1.0,
    #    Emax_clip=+1.0,
    #    Imin_clip=-30e-3,
    #    Imax_clip=+30e-3,
    #    H=H,
    #    W=W,
    #    #blur_iters=cfg.blur_iters,
    #)
    #print(p, cps)
#    plt.imshow(img)
#    plt.show()
    _, E_256 = _resample_to_fixed_n(t_last, E_last, 256)
    _, I_256 = _resample_to_fixed_n(t_last, I_last, 256)
    #print(len(E_256),len(I_256)) 
    return [E_256,I_256], [p,cps]


def generate_split_data(
    num_train: int,
    num_val: int,
    image_size: int = 128,
    add_noise_std: float = 0.0,
    seed: int = 123,
    HER_i0: float|None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    #cfg_tr = SimConfig(image_size=image_size, blur_iters=1, vary_params=True, add_noise_std=add_noise_std, seed=seed)
    #cfg_va = SimConfig(image_size=image_size, blur_iters=1, vary_params=True, add_noise_std=add_noise_std, seed=seed+999)
    cfg_tr = SimConfig(image_size=image_size, vary_params=True, add_noise_std=add_noise_std, seed=seed)
    cfg_va = SimConfig(image_size=image_size, vary_params=True, add_noise_std=add_noise_std, seed=seed+999)
    rng_tr = np.random.default_rng(seed)
    rng_va = np.random.default_rng(seed + 999)
    
    Xtr=[]
    #img_tr=[]
    Ytr=[]
    Xva=[]
    #img_va=[]
    Yva=[]
    for i in range(num_train):
        a,b = simulate_one_cv_image(cfg_tr, rng=rng_tr, index=i, HER_i0=HER_i0)
        Xtr.append(a)
        #img_tr.append(a)
        Ytr.append(b)
    for i in range(num_val):
        a,b=simulate_one_cv_image(cfg_va, rng=rng_va, index=i, HER_i0= HER_i0)
        Xva.append(a)
        #img_va.append(a)
        Yva.append(b)
    return np.asarray(Xtr, dtype=np.float32), np.asarray(Xva, dtype=np.float32),Ytr,Yva


def save_images_to_folder(images:np.array, Emax, Emin, Imax, Imin, out_dir: str, im_size:int, prefix: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        E,I=img
        fig= plt.figure(figsize=(1, 1), dpi=im_size, frameon=False)
        #fig.set_size_inches(im_size,im_size)
#        ax.plot(E,I,'k',linewidth =3)
        ax= fig.add_axes([0., 0., 1., 1.])
        ax.plot(E,I,'k',linewidth =2)
        ax.set_axis_off()
        Emin_clip: float = Emin
        Emax_clip: float = Emax
        Imin_clip: float = Imin
        Imax_clip: float = Imax
        ax.set_xlim(Emin_clip, Emax_clip)
        ax.set_ylim(Imin_clip,Imax_clip)
        #ax.imshow(im_np, aspect='normal')
        #fig.savefig('figure.png', dpi=1)
        out_path = os.path.join(out_dir, f"{prefix}{i:05d}.png")
        fig.savefig(out_path,dpi=im_size,format='jpeg')
#        #img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)  # binary {0,1} -> {0,255}
#        #image8 = Image.fromarray(img8).transpose(Image.FLIP_TOP_BOTTOM)
#        #plt.imshow(img8, interpolation='none')
#        #plt.show()
#        #image8.show()
#        #image8.save(os.path.join(out_dir, f"{prefix}{i:05d}.png"))
    # 1 inch figure at dpi=im_size -> im_size x im_size pixels
#    for i, img in enumerate(images):
#        #print(img.shape)
#        E, I = img
#        #print(E,I)
#        fig = plt.figure(figsize=(1, 1), dpi=im_size)
#        ax = fig.add_axes([0, 0, 1, 1])   # fill the whole canvas
#        ax.axis("off")
#
#        # plot the CV trace
#        ax.plot(E, I, linewidth=2)
#
#        # fixed limits to keep scale consistent
#        ax.set_xlim(-1.0, 1.0)
#        ax.set_ylim(-40e-3, 40e-3)
#
#        # save with no padding/borders
#        out_path = os.path.join(out_dir, f"{prefix}{i:05d}.png")
#        plt.show()
#        fig.savefig(out_path, dpi=im_size, bbox_inches="tight", pad_inches=0)
#        
        plt.close(fig)
    return out_dir

def save_numpy_datasets_images(train_imgs: np.ndarray, val_imgs: np.ndarray, out_dir: str) -> str:
    """
    Save image arrays into out_dir:
      - train_images.npy  (Ntr, H, W) in [0,1]
      - val_images.npy    (Nva, H, W) in [0,1]
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "train_images.npy"), train_imgs.astype(np.float32))
    np.save(os.path.join(out_dir, "val_images.npy"),   val_imgs.astype(np.float32))
    return out_dir

def save_prediction_head(loops, target: list, Emax, Emin, Imax, Imin, out_dir: str, prefix: str)  -> str:
    """ 
    Saves list[p, couples] dict for  
    """
    import os,json
    os.makedirs(out_dir, exist_ok=True)
    value =[]
    for ni,i in enumerate(target):
        ## find E@Imax and E@Imin
        E,I = loops[ni]
        E_at_Imin = all_E_where_I_equals(E,I,Imin)
        E_at_Imax = all_E_where_I_equals(E,I,Imax)

        #print(E_at_Imin, E_at_Imax)

        ### targets: HER/OER overpotentials: E@Imax  and E@Imin, Cdl, HER/OER i0, HER/OER alpha
        p=dict((k,i[0][k]) for k in ('E_max','E_vertex','Cdl') if k in i[0])
        ## Assumed max limit of overpotential without catalyst 
        p['HER_overpot'] = -0.50
        p['OER_overpot'] = 1.5

        if len(E_at_Imin)>0:
            p['HER_overpot'] = min(E_at_Imin) - 0.0   ## Eeq for HER
        if len(E_at_Imax)>0:
            p['OER_overpot'] = max(E_at_Imax) - 1.229  ## Eeq for OER
        p['HER_i0']=i[0]['HER']['i0']
        p['HER_alpha']=i[0]['HER']['alpha']
        p['OER_i0']=i[0]['OER']['i0']
        p['OER_alpha']=i[0]['OER']['alpha']
        #print(p)
        value.append(list(p.values()))
    with open(out_dir+'/'+prefix+'target.txt', 'w') as f:
        f.write(json.dumps(value))
        #print(p,i[0]['HER']['enabled'], i[0]['OER']['enabled'])

def generate_and_save(
    num_train: int = 4096,
    num_val: int = 512,
    image_size: int = 128,
    add_noise_std: float = 0.0,
    seed: int = 24,
    Emax: float = 1.0,
    Emin: float = -0.40,
    Imax: float =  50e-3,
    Imin: float = -50e-3, 
    #np_data: bool = False,
    #image_data: bool = True,
    #prediction_head: bool = False,
    out_npdata_dir: str|None =None,
    save_imgs_dir: str|None = None,   # <--- ADD
    pred_target_dir: str|None = None,
    HER_i0: float|None = None,
    ) -> str:
    Xtr, Xva, Ytr, Yva = generate_split_data(num_train, num_val, image_size=image_size, add_noise_std=add_noise_std, seed=seed, HER_i0=HER_i0)
    if out_npdata_dir:
        save_numpy_datasets_images(Xtr, Xva, out_npdata_dir)
    if save_imgs_dir:
        save_images_to_folder(Xtr, Emax, Emin, Imax, Imin, save_imgs_dir, im_size=image_size, prefix="")
        save_images_to_folder(Xva, Emax, Emin, Imax, Imin, save_imgs_dir, im_size=image_size, prefix="")
    if pred_target_dir:
        save_prediction_head(Xtr, Ytr, Emax, Emin, Imax, Imin, pred_target_dir, prefix ="")   ## sending both [E,I]array and [p,couples] for creating target
        save_prediction_head(Xva, Yva, Emax, Emin, Imax, Imin, pred_target_dir, prefix ="")
        
    #return saved_path
    #return save_numpy_datasets_images(Xtr, Xva, out_dir)

if __name__ == "__main__":
    seed=47
#    path = generate_and_save(num_train=0, num_val=15, image_size=128, out_npdata_dir="./cv_vae2d_data",  save_imgs_dir="./imgs", pred_target_dir='./prediction_target')
    path = generate_and_save(num_train=0, num_val=50, seed=seed, image_size=128, Emax= 1, Emin=-0.3, Imax= 60e-3, Imin=-60e-3, save_imgs_dir="./imgs", pred_target_dir='./pred')
    print("Saved to:", path)


