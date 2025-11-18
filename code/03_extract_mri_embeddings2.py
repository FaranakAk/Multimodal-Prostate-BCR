
# Uses "https://github.com/StandWisdom/MRI-based-Predicted-Transformer-for-Prostate-cancer"'s modality-specific pretrained weights (t2/adc/dwi .pth),
# applies strict masking + intra-mask robust normalization,
# prostate-centered crops, 32 slices/modality,
# and median-pools per-slice 576-d features -> 576*3 = 1,728 per patient.

import os, random, time, numpy as np, pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F




def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(2025)


# ---------------- paths ----------------
ROOT = r"D:\OneDrive - Queen's University\Multimodal-Quiz\Multimodal-Quiz"
PREW = os.path.join(ROOT, "pretrained_weights")  # must contain t2_extractor.pth, adc_extractor.pth, dwi_extractor.pth
OUTD = os.path.join(ROOT, "outputs")
os.makedirs(OUTD, exist_ok=True)

PATHS_CSV   = os.path.join(OUTD, "paths_index.csv")
EMB_CNN_NPY = os.path.join(OUTD, "X_mri_cnn.npy")
IDS_CSV     = os.path.join(OUTD, "X_mri_ids.csv")

# -------------- config -----------------
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SLICES       = 32          # slices per modality
CROP         = 160         # square crop size
STRICT_MASK  = True        # zero outside prostate. normalize using in-mask only
VERBOSE      = True

# -------------- io & utils -------------
def load_nii(path):
    arr = nib.load(path).get_fdata().astype(np.float32)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D volume: {path}")
    # ensure (Z,Y,X)
    z_axis = int(np.argmax(arr.shape))
    if z_axis == 1:  # (X,Z,Y) -> (Z,Y,X)
        arr = np.moveaxis(arr, 1, 0)
        arr = np.moveaxis(arr, 2, 2)
    elif z_axis == 2:  # (Y,X,Z) -> (Z,Y,X)
        arr = np.moveaxis(arr, 2, 0)
    elif z_axis != 0:
        perm = [z_axis] + [i for i in range(3) if i != z_axis]
        arr = np.transpose(arr, perm)
    return arr  # (Z,Y,X)

def robust_norm_masked(slice2d, mask2d, eps=1e-6):
    """Robust 1–99% normalization using only in-mask pixels; outside mask set to 0 if STRICT_MASK."""
    if STRICT_MASK:
        slice2d = np.where(mask2d > 0, slice2d, 0.0)
    vals = slice2d[mask2d > 0]
    if vals.size >= 8:
        p1, p99 = np.percentile(vals, (1, 99))
    else:
        p1, p99 = np.percentile(slice2d, (1, 99))
    if p99 - p1 < eps:
        p99 = p1 + eps
    x = (slice2d - p1) / (p99 - p1)  # [0,1]
    x = x * 2 - 1                    # [-1,1]
    if STRICT_MASK:
        x = np.where(mask2d > 0, x, 0.0)
    return x.astype(np.float32)

def pick_mask_spanning_indices(msk3d, n=SLICES):
    Z = msk3d.shape[0]
    fg = np.where(msk3d.sum(axis=(1, 2)) > 0)[0]
    if fg.size == 0:
        start = max(0, Z // 2 - n // 2)
        idx = np.arange(start, min(Z, start + n))
    else:
        zmin, zmax = int(fg.min()), int(fg.max())
        if zmax == zmin:
            idx = np.full(n, zmin, dtype=int)
        else:
            idx = np.round(np.linspace(zmin, zmax, n)).astype(int)
    if idx.size < n:
        idx = np.pad(idx, (0, n - idx.size), mode="edge")
    return np.clip(idx, 0, Z - 1)

def prostate_center_from_mask(msk3d):
    proj = (msk3d > 0).sum(axis=0)  # (Y,X)
    ys, xs = np.where(proj > 0)
    if ys.size == 0:
        return None
    return int(np.median(ys)), int(np.median(xs))

def crop_center(slice2d, cy, cx, size=CROP):
    H, W = slice2d.shape
    y1 = max(cy - size // 2, 0); y2 = min(cy + size // 2, H)
    x1 = max(cx - size // 2, 0); x2 = min(cx + size // 2, W)
    out = np.zeros((size, size), dtype=np.float32)
    oy = (size - (y2 - y1)) // 2
    ox = (size - (x2 - x1)) // 2
    out[oy:oy + (y2 - y1), ox:ox + (x2 - x1)] = slice2d[y1:y2, x1:x2]
    return out

def make_stack(vol, msk, n=SLICES, size=CROP):
    """Return (n, 3, size, size) masked & normalized slices."""
    idx = pick_mask_spanning_indices(msk, n=n)
    ctr = prostate_center_from_mask(msk)
    if ctr is None:
        ctr = (vol.shape[1] // 2, vol.shape[2] // 2)
    cy, cx = ctr
    out = []
    for z in idx:
        sl  = vol[z]
        msl = (msk[z] > 0).astype(np.uint8)
        sln = robust_norm_masked(sl, msl)  # [-1,1], outside→0 if STRICT_MASK
        crp = crop_center(sln, cy, cx, size=size)
        out.append(np.stack([crp, crp, crp], axis=0))  # 3 channels for MobileNet
    return np.stack(out, axis=0)  # (n,3,H,W)

# -------------- pretrained backbone -------------
class MobileNetSmallFeatures(nn.Module):
    """Torchvision MobileNetV3-Small 'features' block (no classifier)."""
    def __init__(self):
        super().__init__()
        from torchvision import models
        m = models.mobilenet_v3_small(weights=None)
        self.features = m.features
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # (N,3,H,W)
        x = self.features(x)          # (N,576,h,w)
        x = self.relu(x)
        return x

def build_extractor(pth_path):
    """
    Load a state_dict from the repo into MobileNetV3-Small features.
    We then GAP to 576-d per slice.
    """
    sd = torch.load(pth_path, map_location=DEVICE)
    backbone = MobileNetSmallFeatures().to(DEVICE).eval()
    # Accept state_dict or full module.state_dict()
    state = sd.state_dict() if isinstance(sd, nn.Module) else sd
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] Loading {os.path.basename(pth_path)} with strict=False. "
              f"Missing={len(missing)}, Unexpected={len(unexpected)}")
    @torch.no_grad()
    def run(batch3x):
        feat = backbone(batch3x)                           # (N,576,h,w)
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)  # (N,576)
        return feat
    return run

# ---------------- main -----------------
def main():
    # sanity check pretrained files exist
    need = ["t2_extractor.pth", "adc_extractor.pth", "dwi_extractor.pth"]
    for f in need:
        fp = os.path.join(PREW, f)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing pretrained file: {fp}")

    dfp = pd.read_csv(PATHS_CSV, dtype={"patient_id": str}).sort_values("patient_id")

    # build modality-specific extractors from repo weights
    run_t2  = build_extractor(os.path.join(PREW, "t2_extractor.pth"))
    run_adc = build_extractor(os.path.join(PREW, "adc_extractor.pth"))
    run_hbv = build_extractor(os.path.join(PREW, "dwi_extractor.pth"))  # HBV (a.k.a. high-b-value DWI)

    ids, embs = [], []

    with torch.no_grad():
        for i, r in enumerate(dfp.itertuples(index=False), 1):
            t0 = time.time()
            pid = r.patient_id
            t2  = load_nii(r.t2w)
            adc = load_nii(r.adc)
            hbv = load_nii(r.hbv)
            msk = (load_nii(r.mask_t2w) > 0).astype(np.uint8)

            # stacks: (N,3,CROP,CROP)
            t2_s  = torch.from_numpy(make_stack(t2,  msk, n=SLICES, size=CROP)).to(DEVICE)
            adc_s = torch.from_numpy(make_stack(adc, msk, n=SLICES, size=CROP)).to(DEVICE)
            hbv_s = torch.from_numpy(make_stack(hbv, msk, n=SLICES, size=CROP)).to(DEVICE)

            # slice-level features -> median over slices
            v_t2  = run_t2(t2_s).median(dim=0).values.cpu().numpy()   # (576,)
            v_adc = run_adc(adc_s).median(dim=0).values.cpu().numpy() # (576,)
            v_hbv = run_hbv(hbv_s).median(dim=0).values.cpu().numpy() # (576,)

            vec = np.concatenate([v_t2, v_adc, v_hbv], axis=0).astype(np.float32)  # (1728,)
            ids.append(pid)
            embs.append(vec)

            if VERBOSE:
                print(f"[{i}/{len(dfp)}] {pid}: embed={vec.shape}  time={time.time()-t0:.2f}s", flush=True)

    X = np.vstack(embs)
    np.save(EMB_CNN_NPY, X)
    pd.DataFrame({"patient_id": ids}).to_csv(IDS_CSV, index=False)
    print(f"Saved embeddings: {EMB_CNN_NPY} {X.shape}")
    print(f"IDs saved: {IDS_CSV}")
    print("Done on", DEVICE)

if __name__ == "__main__":
    print("Using device:", DEVICE)
    main()
