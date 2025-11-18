import os, random, json, glob, math, numpy as np, pandas as pd
from typing import List, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import wandb

import warnings
warnings.filterwarnings('ignore')


# ---- Reproducibility ----
SEED = 1337  

def seed_everything(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def _seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---- Paths / globals ----
ROOT = r"D:\OneDrive - Queen's University\Multimodal-Quiz\Multimodal-Quiz"
OUTD = os.path.join(ROOT, "outputs")
BAGS_IDX = os.path.join(OUTD, "mri_bags_index.csv")
FOLDS_CSV = os.path.join(ROOT, "data_split_5fold.csv")
CLIN_DIR  = os.path.join(ROOT, "clinical_data")

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS  = 50
PATIENCE= 12
LR      = 1e-3
WD      = 1e-4
H_INST  = 384     # instance encoder dim (MRI slice embedding → hidden)
H_CLS   = 32      # clinical MLP hidden/output dim
D_FEAT  = 1728    # per-slice feature size from 03
D_CLS   = None    # will be set after clinical preprocessing


EXPERIMENT_TAG = "meanpool"  # e.g. "full", "noclin", "meanpool"

BATCH_SIZE = 8          
SAVE_ATTN  = True       # save attention weights for test patients
ATTN_DIR   = os.path.join(OUTD, f"attn_{EXPERIMENT_TAG}")
os.makedirs(ATTN_DIR, exist_ok=True)
CKPT_DIR   = os.path.join(OUTD, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


WANDB_PROJECT = "prostate_multimodal"
WANDB_ENTITY = None  


# ABMIL ablation switches
ABLATE_NO_CLIN = False      # if True, ignore clinical branch (MRI-only MIL)
ABLATE_NO_POS = False       # if True, ignore z_idx/mod positional encodings
ABLATE_MEAN_POOL = True    # if True, replace attention with simple mean pooling

# Optional single-modality filter for bags (None, "t2", "adc", or "hbv")
MOD_FILTER = None
MOD_MAP = {"t2": 0, "adc": 1, "hbv": 2}


# ---------- metrics ----------
def harrell_cindex_risk(events: np.ndarray, times: np.ndarray, risks: np.ndarray) -> float:
    n = len(times)
    idx = np.argsort(times, kind="mergesort")
    times = times[idx]; events = events[idx]; risks = risks[idx]
    conc = 0.0; perm = 0.0
    for i in range(n-1):
        if not events[i]:
            continue
        for j in range(i+1, n):
            if times[j] <= times[i]:
                continue
            perm += 1
            if risks[i] > risks[j]: conc += 1
            elif risks[i] == risks[j]: conc += 0.5
    return np.nan if perm == 0 else conc/perm


# ---------- clinical design ----------
def load_clinical_table() -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(CLIN_DIR, "*.json")):
        pid = os.path.splitext(os.path.basename(path))[0]
        d = json.load(open(path, "r"))

        rec = {
            "patient_id": pid,
            "age_at_prostatectomy": d.get("age_at_prostatectomy"),
            "primary_gleason":      d.get("primary_gleason"),
            "secondary_gleason":    d.get("secondary_gleason"),
            "tertiary_gleason":     d.get("tertiary_gleason"),
            "ISUP":                 d.get("ISUP"),
            "pre_operative_PSA":    d.get("pre_operative_PSA"),
            "positive_surgical_margins": d.get("positive_surgical_margins"),
        }
        for k in ["positive_lymph_nodes","capsular_penetration","invasion_seminal_vesicles","lymphovascular_invasion"]:
            v = d.get(k)
            if v in (None, "", "x", "X"):
                rec[k] = np.nan
            else:
                try:
                    rec[k] = int(float(v))
                except Exception:
                    rec[k] = np.nan

        rec["pT_stage"]       = d.get("pT_stage")
        rec["earlier_therapy"]= d.get("earlier_therapy")

        bcr = d.get("BCR")
        rec["event"] = 1 if str(bcr) in ("1","1.0","true","True") or bcr==1 else 0
        try:
            rec["time"] = float(d.get("time_to_follow-up/BCR"))
        except Exception:
            rec["time"] = np.nan
        rows.append(rec)

    df = pd.DataFrame(rows)
    for leak in ["BCR_PSA","BCR","time_to_follow-up/BCR"]:
        if leak in df.columns:
            df = df.drop(columns=[leak])
    return df


def build_clin_design(df):
    numeric = ["age_at_prostatectomy","primary_gleason","secondary_gleason","tertiary_gleason",
               "ISUP","pre_operative_PSA"]
    binary = ["positive_surgical_margins","positive_lymph_nodes","capsular_penetration",
              "invasion_seminal_vesicles","lymphovascular_invasion"]
    cat    = ["pT_stage","earlier_therapy"]

    df = df.copy()
    for c in binary:
        if c not in df.columns:
            continue
        df[c] = df[c].apply(lambda x: 1 if str(x) in ("1","1.0") else 0 if str(x) in ("0","0.0") else np.nan)

    for c in cat:
        if c in df.columns:
            df[c] = df[c].astype("category")

    X_num = df[numeric].astype(float)
    imp = SimpleImputer(strategy="median")
    X_num_imp = imp.fit_transform(X_num)

    X_bin = df[binary].astype(float)
    X_bin = X_bin.fillna(0.0).values

    X_cat_list = []
    for c in cat:
        if c in df.columns:
            c_codes = df[c].cat.codes.replace(-1, np.nan).values
            c_codes = c_codes.reshape(-1,1)
            X_cat_list.append(c_codes)
    X_cat = np.concatenate(X_cat_list, axis=1) if X_cat_list else np.zeros((len(df),0))

    X = np.concatenate([X_num_imp, X_bin, X_cat], axis=1)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    return Xs, {"numeric": numeric, "binary": binary, "cat": cat}


# ---------- Dataset ----------
class BagDataset(Dataset):
    def __init__(self, table_df):
        self.df = table_df.reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        npz = np.load(r["bag_path"])
        X = npz["X"].astype(np.float32)  # (S, D_FEAT)

        z_idx = npz["z_idx"].astype(np.int32) if "z_idx" in npz else np.arange(len(X), dtype=np.int32)
        mod   = npz["mod"].astype(np.int32)   if "mod" in npz else np.zeros(len(X), dtype=np.int32)

        if MOD_FILTER is not None:
            m_id = MOD_MAP.get(MOD_FILTER, None)
            if m_id is not None:
                mask = (mod == m_id)
                if np.any(mask):
                    X = X[mask]
                    z_idx = z_idx[mask]
                    mod = mod[mask]

        clin = r["clin_vec"].astype(np.float32)
        y_event = bool(int(r["event"]))
        y_time  = float(r["time"])
        return X, clin, y_event, y_time, r["patient_id"], z_idx, mod


def collate_bags(batch):
    Xs, Cs, Es, Ts, IDs, Zs, Ms = zip(*batch)
    return Xs, np.stack(Cs), np.array(Es, dtype=bool), np.array(Ts, dtype=float), list(IDs), Zs, Ms


# ---------- model ----------
class InstanceEncoder(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_hidden), nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class GatedAttention(nn.Module):
    def __init__(self, d_in, d_att=128, temperature=1.2):
        super().__init__()
        self.V = nn.Linear(d_in, d_att, bias=True)
        self.U = nn.Linear(d_in, d_att, bias=True)
        self.w = nn.Linear(d_att, 1, bias=True)
        self.temperature = temperature

        for m in [self.V, self.U, self.w]:
            if hasattr(m, "weight"):
                nn.init.kaiming_uniform_(m.weight, a=1.0)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 1e-3)

    def forward(self, M):
        v = torch.tanh(self.V(M))
        u = torch.sigmoid(self.U(M))
        a = self.w(v * u).squeeze(-1)
        a = a * self.temperature
        w = torch.softmax(a, dim=0)
        H = torch.sum(w.unsqueeze(-1) * M, dim=0)
        return H, w, a


class ClinEncoder(nn.Module):
    def __init__(self, d_in: int, d_out: int, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_out), nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
    def forward(self, x):
        return self.net(x)


class ABMILCox(nn.Module):
    def __init__(self, d_feat: int, d_hidden: int, d_clin_in: int, d_clin_out: int):
        super().__init__()
        self.inst = InstanceEncoder(d_in=d_feat, d_hidden=d_hidden)
        self.attn = GatedAttention(d_hidden, d_att=128, temperature=1.2)

        self.pos_proj = nn.Sequential(
            nn.Linear(4, d_hidden), nn.Tanh()
        )

        self.clin = ClinEncoder(d_in=d_clin_in, d_out=d_clin_out)
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_hidden + d_clin_out),
            nn.Linear(d_hidden + d_clin_out, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.risk = nn.Linear(d_hidden, 1)
        self.last_attn_weights = []

    def forward(self, X, clin_vec, z_idx=None, mod=None):
        M = self.inst(X)

        if (not ABLATE_NO_POS) and (z_idx is not None) and (mod is not None):
            S = M.shape[0]
            if S > 0:
                z = z_idx.float()
                z = (z - z.min()) / (z.max().clamp(min=1) - z.min())
                z = z.view(S, 1)
                mo = torch.zeros(S, 3, device=M.device)
                mo[torch.arange(S), mod.clamp(0, 2).long()] = 1.0
                P = torch.cat([z, mo], dim=1)
                M = M + self.pos_proj(P)

        if ABLATE_MEAN_POOL:
            if M.shape[0] == 0:
                H = torch.zeros(M.shape[1], device=M.device)
                w = torch.zeros(0, device=M.device)
            else:
                H = M.mean(dim=0)
                w = torch.full((M.shape[0],), 1.0 / M.shape[0], device=M.device)
        else:
            H, w, _ = self.attn(M)
            if self.training:
                self.last_attn_weights.append(w)

        if ABLATE_NO_CLIN:
            Z = H
        else:
            c = self.clin(clin_vec)
            Z = torch.cat([H, c], dim=0)
            Z = self.fuse(Z)

        risk = self.risk(Z).squeeze()
        return risk, w


def cox_ph_loss(risks: torch.Tensor, events: torch.Tensor, times: torch.Tensor):
    order = torch.argsort(times, descending=True)
    r = risks[order]; e = events[order]
    lse = torch.logcumsumexp(r, dim=0)
    loglik = (r - lse) * e
    return -loglik[e.bool()].sum() / (e.sum().clamp(min=1.0))


# ---------- training ----------
def train_fold(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, wd=WD, fold_idx: int = 0):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    warmup = max(5, epochs // 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup))
    best_val = float("inf")
    patience = 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []

        for Xs, Cs, Es, Ts, _, Zs, Ms in train_loader:
            risks = []
            events = []
            times = []
            for i in range(len(Xs)):
                X = torch.from_numpy(Xs[i]).to(DEVICE)
                C = torch.from_numpy(Cs[i]).to(DEVICE)
                Zi = torch.from_numpy(Zs[i]).to(DEVICE)
                Mi = torch.from_numpy(Ms[i]).to(DEVICE)
                r, _ = model(X, C, Zi, Mi)
                risks.append(r.unsqueeze(0))
                events.append(float(Es[i]))
                times.append(float(Ts[i]))

            risks = torch.cat(risks, dim=0)
            events = torch.tensor(events, dtype=torch.float32, device=DEVICE)
            times = torch.tensor(times, dtype=torch.float32, device=DEVICE)

            risks = risks - risks.mean()

            loss = cox_ph_loss(risks, events, times)

            ENT_LAMBDA = 5e-4
            if hasattr(model, "last_attn_weights") and model.last_attn_weights:
                ent = 0.0
                for w in model.last_attn_weights:
                    w_safe = torch.clamp(w, 1e-8, 1.0)
                    ent = ent - torch.sum(w_safe * torch.log(w_safe))
                ent = ent / len(model.last_attn_weights)
                loss = loss + ENT_LAMBDA * ent

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            scheduler.step()
            model.last_attn_weights = []

            tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_losses = []
            for Xs, Cs, Es, Ts, _, Zs, Ms in val_loader:
                risks = []
                events = []
                times = []
                for i in range(len(Xs)):
                    X = torch.from_numpy(Xs[i]).to(DEVICE)
                    C = torch.from_numpy(Cs[i]).to(DEVICE)
                    Zi = torch.from_numpy(Zs[i]).to(DEVICE)
                    Mi = torch.from_numpy(Ms[i]).to(DEVICE)
                    r, _ = model(X, C, Zi, Mi)
                    risks.append(r.unsqueeze(0))
                    events.append(float(Es[i]))
                    times.append(float(Ts[i]))

                risks = torch.cat(risks, dim=0)
                events = torch.tensor(events, dtype=torch.float32, device=DEVICE)
                times = torch.tensor(times, dtype=torch.float32, device=DEVICE)
                risks = risks - risks.mean()

                vloss = cox_ph_loss(risks, events, times).item()
                val_losses.append(vloss)

            v = np.mean(val_losses) if len(val_losses) else np.inf

        mean_tr = float(np.mean(tr_losses)) if tr_losses else float("inf")
        print(f"  epoch {ep:02d} | train_loss={mean_tr:.4f} | val_loss={v:.4f}")

        wandb.log(
            {
                "fold": int(fold_idx),
                "epoch": ep,
                "train_loss": mean_tr,
                "val_loss": float(v),
            }
        )

        if v < best_val - 1e-4:
            best_val = v
            patience = 0
            best_state = {
                k: (vv.detach().cpu().clone() if isinstance(vv, torch.Tensor) else vv)
                for k, vv in model.state_dict().items()
            }
        else:
            patience += 1
            if patience >= PATIENCE:
                print("  early stop.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------- main ----------
def main():
    seed_everything(SEED)
    os.makedirs(OUTD, exist_ok=True)

    bags_df = pd.read_csv(BAGS_IDX, dtype={"patient_id": str})
    df_clin_raw = load_clinical_table()
    folds = pd.read_csv(FOLDS_CSV, dtype={"patient_id": str})

    Xclin_all, _ = build_clin_design(df_clin_raw)
    meta_clin = df_clin_raw[["patient_id", "event", "time"]].copy()
    meta_clin["clin_vec"] = list(Xclin_all.astype(np.float32))

    global D_CLS
    D_CLS = Xclin_all.shape[1]

    meta = bags_df.merge(meta_clin, on="patient_id").merge(folds, on="patient_id")
    print(f"Using device: {DEVICE}")
    print(f"Feature blocks prepared. N={len(meta)} | MRI=({D_FEAT}) per slice | Clin={D_CLS}")

    scores = []
    all_rows = []

    for k in sorted(meta["fold"].unique()):
        tr_all = meta[meta["fold"] != k].reset_index(drop=True)
        tr_df, va_df = train_test_split(
            tr_all, test_size=0.2, random_state=SEED, stratify=tr_all["event"]
        )
        te_df = meta[meta["fold"] == k].reset_index(drop=True)

        ds_tr = BagDataset(tr_df)
        ds_va = BagDataset(va_df)
        ds_te = BagDataset(te_df)

        g = torch.Generator()
        g.manual_seed(SEED)
        dl_tr = DataLoader(
            ds_tr,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_bags,
            worker_init_fn=_seed_worker,
            generator=g,
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_bags,
            worker_init_fn=_seed_worker,
            generator=g,
        )
        dl_te = DataLoader(
            ds_te,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_bags,
            worker_init_fn=_seed_worker,
            generator=g,
        )

        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=f"abmil_{EXPERIMENT_TAG}_fold_{k}",
            config={
                "epochs": EPOCHS,
                "lr": LR,
                "weight_decay": WD,
                "H_INST": H_INST,
                "H_CLS": H_CLS,
                "D_FEAT": D_FEAT,
                "D_CLS": D_CLS,
                "ABLATE_NO_CLIN": ABLATE_NO_CLIN,
                "ABLATE_NO_POS": ABLATE_NO_POS,
                "ABLATE_MEAN_POOL": ABLATE_MEAN_POOL,
                "MOD_FILTER": MOD_FILTER,
                "EXPERIMENT_TAG": EXPERIMENT_TAG,
            },
            reinit=True,
        )

        model = ABMILCox(
            d_feat=D_FEAT,
            d_hidden=H_INST,
            d_clin_in=D_CLS,
            d_clin_out=H_CLS,
        ).to(DEVICE)
        if k == 0:
            print(
                f"[diag] dims → d_feat={D_FEAT}, H_INST={H_INST}, d_clin_in={D_CLS}, d_clin_out={H_CLS}"
            )

        print(f"\n[fold {k}] train={len(tr_df)} val={len(va_df)} test={len(te_df)}")
        model = train_fold(model, dl_tr, dl_va, epochs=EPOCHS, lr=LR, wd=WD, fold_idx=int(k))

        # save checkpoint for this fold (for HuggingFace upload or later reuse)
        ckpt_path = os.path.join(CKPT_DIR, f"abmil_{EXPERIMENT_TAG}_fold{k}.pth")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "fold": int(k),
                "config": {
                    "D_FEAT": D_FEAT,
                    "H_INST": H_INST,
                    "D_CLS": D_CLS,
                    "H_CLS": H_CLS,
                    "ABLATE_NO_CLIN": ABLATE_NO_CLIN,
                    "ABLATE_NO_POS": ABLATE_NO_POS,
                    "ABLATE_MEAN_POOL": ABLATE_MEAN_POOL,
                    "MOD_FILTER": MOD_FILTER,
                },
            },
            ckpt_path,
        )
        print(f"[info] Saved ABMIL checkpoint for fold {k} to {ckpt_path}")

        # ------- test C-index + attention saving -------
        risks, evts, times = [], [], []
        model.eval()
        with torch.no_grad():
            fold_dir = os.path.join(ATTN_DIR, f"fold_{k}")
            if SAVE_ATTN:
                os.makedirs(fold_dir, exist_ok=True)

            for Xs, Cs, Es, Ts, IDs, Zs, Ms in dl_te:
                for i in range(len(Xs)):
                    pid = IDs[i]
                    X = torch.from_numpy(Xs[i]).to(DEVICE)
                    C = torch.from_numpy(Cs[i]).to(DEVICE)
                    Zi = torch.from_numpy(Zs[i]).to(DEVICE)
                    Mi = torch.from_numpy(Ms[i]).to(DEVICE)
                    r, w = model(X, C, Zi, Mi)

                    risk_val = float(r.detach().cpu())
                    event_val = bool(Es[i])
                    time_val = float(Ts[i])

                    risks.append(risk_val)
                    evts.append(event_val)
                    times.append(time_val)

                    all_rows.append(
                        {
                            "patient_id": pid,
                            "fold": int(k),
                            "event": int(event_val),
                            "time": time_val,
                            "risk_abmil": risk_val,
                            "experiment": EXPERIMENT_TAG,
                        }
                    )

                    if SAVE_ATTN:
                        bag_path = meta.loc[meta["patient_id"] == pid, "bag_path"].values[0]
                        meta_dict = {}
                        try:
                            with np.load(bag_path) as zf:
                                if "z_idx" in zf:
                                    meta_dict["z_idx"] = zf["z_idx"].tolist()
                                if "mod" in zf:
                                    meta_dict["mod"] = zf["mod"].tolist()
                        except Exception:
                            pass

                        attn = w.detach().cpu().numpy()
                        out_csv = os.path.join(fold_dir, f"{pid}_attn.csv")
                        rows = []
                        for row_idx, w_row in enumerate(attn):
                            row = {"row_in_bag": int(row_idx), "weight": float(w_row)}
                            if "z_idx" in meta_dict and row_idx < len(meta_dict["z_idx"]):
                                row["z_idx"] = int(meta_dict["z_idx"][row_idx])
                            if "mod" in meta_dict and row_idx < len(meta_dict["mod"]):
                                row["mod"] = int(meta_dict["mod"][row_idx])
                            rows.append(row)
                        pd.DataFrame(rows).to_csv(out_csv, index=False)

                        topk = np.argsort(-attn)[:5]
                        # print(
                        #     f"  [attn] {pid} top5 rows:",
                        #     [(int(t), float(attn[t])) for t in topk],
                        # )

        cidx = harrell_cindex_risk(
            np.array(evts, bool),
            np.array(times, float),
            np.array(risks, float),
        )
        print(f"[fold {k}] C-index = {cidx:.3f}")
        scores.append(cidx)

        wandb.log({"fold": int(k), "test_cindex": float(cidx)})
        run.finish()

    print(
        "ABMIL | C-index per fold:",
        np.round(scores, 3),
        f"| mean={np.nanmean(scores):.3f}±{np.nanstd(scores):.3f}",
    )

    if all_rows:
        pred_df = pd.DataFrame(all_rows)
        out_pred = os.path.join(OUTD, f"abmil_casewise_predictions_{EXPERIMENT_TAG}.csv")
        pred_df.to_csv(out_pred, index=False)
        print(f"[info] Saved case-wise ABMIL predictions to {out_pred}")


if __name__ == "__main__":
    main()
