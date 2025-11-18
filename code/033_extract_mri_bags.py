
import os, glob, numpy as np, pandas as pd, nibabel as nib
import torch, torch.nn as nn, torch.nn.functional as F

ROOT = r"D:\OneDrive - Queen's University\Multimodal-Quiz\Multimodal-Quiz"
PREW = os.path.join(ROOT, "pretrained_weights")
OUTD = os.path.join(ROOT, "outputs")
os.makedirs(OUTD, exist_ok=True)

PATHS_CSV = os.path.join(OUTD, "paths_index.csv")
BAGS_DIR  = os.path.join(OUTD, "mri_bags")
os.makedirs(BAGS_DIR, exist_ok=True)
BAGS_IDX  = os.path.join(OUTD, "mri_bags_index.csv")

# ---- config ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SLICE_COUNT = 32
CROP = 160
APPLY_MASK = True          # zero background outside prostate
STRICT_INMASK_NORM = True  # robust scaling from masked pixels only

def load_nii(path):
    return nib.load(path).get_fdata().astype(np.float32)  # (Z, Y, X)

def pick_mask_slices(msk3d, n=SLICE_COUNT):
    z_present = np.where(msk3d.sum(axis=(1,2)) > 0)[0]
    if len(z_present) == 0:
        Z = msk3d.shape[0]
        start = max(0, Z//2 - n//2)
        idx = np.arange(start, min(Z, start+n))
        if len(idx) < n: idx = np.pad(idx, (0, n-len(idx)), mode='edge')
        return idx
    zmin, zmax = int(z_present.min()), int(z_present.max())
    if zmax == zmin:
        return np.clip(np.full(n, zmin), 0, msk3d.shape[0]-1)
    return np.round(np.linspace(zmin, zmax, n)).astype(int)

def crop_around_center(sl, cy, cx, size=CROP):
    H, W = sl.shape
    y1 = max(cy - size//2, 0); y2 = min(cy + size//2, H)
    x1 = max(cx - size//2, 0); x2 = min(cx + size//2, W)
    patch = np.zeros((size, size), dtype=np.float32)
    oy1 = (size - (y2 - y1)) // 2
    ox1 = (size - (x2 - x1)) // 2
    patch[oy1:oy1+(y2-y1), ox1:ox1+(x2-x1)] = sl[y1:y2, x1:x2]
    return patch

def robust_norm_from_mask(patch, patch_msk, eps=1e-6):
    if STRICT_INMASK_NORM and patch_msk.sum() > 10:
        vals = patch[patch_msk > 0]
    else:
        vals = patch.reshape(-1)
    p1, p99 = np.percentile(vals, (1, 99))
    if p99 <= p1: p99 = p1 + eps
    x = (patch - p1) / (p99 - p1)
    x = x * 2 - 1
    return x

def make_slices(vol3d, msk3d, n=SLICE_COUNT, size=CROP, apply_mask=True):
    idx = pick_mask_slices(msk3d, n=n)
    ys, xs = np.where(msk3d.sum(axis=0) > 0)
    cy = int(np.median(ys)) if len(ys) else vol3d.shape[1]//2
    cx = int(np.median(xs)) if len(xs) else vol3d.shape[2]//2

    imgs, msks = [], []
    for z in idx:
        sl = vol3d[z]
        sl_m = (msk3d[z] > 0).astype(np.uint8)
        p   = crop_around_center(sl, cy, cx, size=size)
        pm  = crop_around_center(sl_m, cy, cx, size=size)
        x   = robust_norm_from_mask(p, pm)
        if apply_mask:
            x = x * pm
        # 3 channels as expected by torchvision convs
        imgs.append(np.stack([x, x, x], axis=0))   # (3,H,W)
        msks.append(pm)
    return np.stack(imgs, axis=0), np.stack(msks, axis=0)  # (N,3,H,W), (N,H,W)

# ---- StandWisdom backbone ----
class MobileNetSmallFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        m = models.mobilenet_v3_small(weights=None)
        self.features = m.features
    def forward(self, x):  # x: (B,3,H,W)
        return self.features(x)  # (B,576,h,w)

def load_standwisdom_backbone(pth_path):
    sd = torch.load(pth_path, map_location="cpu")
    if isinstance(sd, nn.Module):
        model = sd.to(DEVICE).eval()
        return model, None
    backbone = MobileNetSmallFeatures().to(DEVICE).eval()
    missing, unexpected = backbone.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] {os.path.basename(pth_path)} strict=False: Missing={len(missing)} Unexpected={len(unexpected)}")
    return backbone, None

@torch.no_grad()
def extract_feats_576(backbone, x3):  # x3: (N,3,H,W) float32 [-1,1]
    x = torch.from_numpy(x3).to(DEVICE)
    f = backbone(x)                                  # (N,576,h,w)
    f = F.adaptive_avg_pool2d(f, (1,1)).squeeze(-1).squeeze(-1)  # (N,576)
    return f.detach().cpu().numpy()

def main():
    assert os.path.exists(PATHS_CSV), f"Missing {PATHS_CSV}"
    dfp = pd.read_csv(PATHS_CSV, dtype={"patient_id": str}).sort_values("patient_id")

    # load StandWisdom extractors
    t2_bp, _  = load_standwisdom_backbone(os.path.join(PREW, "t2_extractor.pth"))
    adc_bp, _ = load_standwisdom_backbone(os.path.join(PREW, "adc_extractor.pth"))
    dwi_bp, _ = load_standwisdom_backbone(os.path.join(PREW, "dwi_extractor.pth"))

    rows = []
    for i, r in enumerate(dfp.itertuples(index=False), 1):
        pid = r.patient_id
        t2  = load_nii(r.t2w)
        adc = load_nii(r.adc)
        hbv = load_nii(r.hbv)
        msk = (load_nii(r.mask_t2w) > 0).astype(np.uint8)

        # build modality-aligned slice stacks
        t2_s,  _ = make_slices(t2,  msk, n=SLICE_COUNT, size=CROP, apply_mask=APPLY_MASK)
        adc_s, _ = make_slices(adc, msk, n=SLICE_COUNT, size=CROP, apply_mask=APPLY_MASK)
        hbv_s, _ = make_slices(hbv, msk, n=SLICE_COUNT, size=CROP, apply_mask=APPLY_MASK)

        # per-slice features (576 each) → concat to 1728
        f_t2  = extract_feats_576(t2_bp,  t2_s)   # (S,576)
        f_adc = extract_feats_576(adc_bp, adc_s)
        f_hbv = extract_feats_576(dwi_bp, hbv_s)

        bag = np.concatenate([f_t2, f_adc, f_hbv], axis=1)  # (S, 1728)
        out_path = os.path.join(BAGS_DIR, f"{pid}.npz")
        np.savez_compressed(out_path, X=bag.astype(np.float32), patient_id=pid)

        rows.append({"patient_id": pid, "bag_path": out_path, "n_slices": bag.shape[0], "feat_dim": bag.shape[1]})
        if i % 10 == 0:
            print(f"[{i}/{len(dfp)}] saved {pid}: {bag.shape}")

    pd.DataFrame(rows).to_csv(BAGS_IDX, index=False)
    print("Saved bags index:", BAGS_IDX)

if __name__ == "__main__":
    print("Using device:", DEVICE)
    main()
