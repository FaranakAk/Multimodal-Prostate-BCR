
# Non-linear survival baselines on MRI/Clin: RSF, GBSA, CGB, FastSurvivalSVM, Coxnet, + DeepSurv
# Evaluates MRI-only (with PCA), Clin-only, and MRI+Clin.

import os, random, json, glob, warnings
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit

from sksurv.metrics import concordance_index_censored
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxnetSurvivalAnalysis
import torch

import warnings
warnings.filterwarnings('ignore')


def set_seed(seed=2025):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(2025)


# 
HAS_CGB = True
try:
    from sksurv.linear_model import ComponentwiseGradientBoostingSurvivalAnalysis
except Exception:
    HAS_CGB = False

HAS_SVM = True
try:
    
    from sksurv.svm import FastSurvivalSVM
except Exception:
    HAS_SVM = False


HAS_DEEPSURV = True
try:
    import torch
    import torch.nn as nn
    import torchtuples as tt
    from pycox.models import CoxPH
except Exception:
    HAS_DEEPSURV = False

# ---------------- Config ----------------
ROOT = r"D:\OneDrive - Queen's University\Multimodal-Quiz\Multimodal-Quiz"

OUTD     = os.path.join(ROOT, "outputs")
EMB_PATH = os.path.join(OUTD, "X_mri_vit.npy")
IDS_PATH = os.path.join(OUTD, "X_mri_ids.csv")
FOLDS_CSV = os.path.join(ROOT, "data_split_5fold.csv")
CLIN_DIR  = os.path.join(ROOT, "clinical_data")

RUN_MRI_ONLY       = True
RUN_CLIN_ONLY      = True
RUN_MRI_PLUS_CLIN  = True

# Toggle models
USE_RSF       = True
USE_GBSA      = True
USE_CGB       = True and HAS_CGB
USE_SVM       = True and HAS_SVM
USE_COX       = True   
USE_DEEPSURV  = True and HAS_DEEPSURV

# MRI PCA grid 
PCA_GRID = [32]  

# Inner-CV
INNER_SPLITS = 2
INNER_TEST   = 0.25
RANDOM_STATE = 0


RSF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    bootstrap=True
)


GBSA_PARAMS = dict(
    learning_rate=0.05,
    n_estimators=400,
    max_depth=2,
    subsample=0.9,
    random_state=RANDOM_STATE
)


CGB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.1,
    random_state=RANDOM_STATE
)


SVM_ALPHAS = np.logspace(-3, 1, 5)


ALPHAS_RIDGE = np.logspace(-2, 2, 15)
ALPHAS_ELNET = np.logspace(-2, 1, 15)
L1R_MRI    = 1e-6
L1R_CLIN   = 1e-6
L1R_CONCAT = 0.25


DEEPSURV_PARAMS = dict(
    hidden_layers=[128, 64],
    dropout=0.10,
    lr=1e-3,
    weight_decay=1e-4,     
    batch_size=32,
    epochs=200,            
    patience=20,           
    num_workers=0,         
    seed=RANDOM_STATE
)

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(precision=3, suppress=True)

# --------------- Utilities ---------------
def load_embeddings():
    X_mri = np.load(EMB_PATH)
    ids   = pd.read_csv(IDS_PATH, dtype={'patient_id': str})
    if X_mri.shape[0] != len(ids):
        raise RuntimeError("Embeddings and IDs length mismatch.")
    print(f"Using CNN embeddings: {X_mri.shape}")
    return X_mri, ids

def build_clinical_table():
    rows = []
    coerced_bin = 0
    for path in glob.glob(os.path.join(CLIN_DIR, "*.json")):
        pid = os.path.splitext(os.path.basename(path))[0]
        d = json.load(open(path, "r"))
        rec = {"patient_id": pid}

        # numeric-ish (coerce)
        for k in ["age_at_prostatectomy","primary_gleason","secondary_gleason",
                  "tertiary_gleason","ISUP","pre_operative_PSA","positive_surgical_margins"]:
            rec[k] = pd.to_numeric(d.get(k), errors="coerce")

        # binary (0/1)
        for k in ["positive_lymph_nodes","capsular_penetration",
                  "invasion_seminal_vesicles","lymphovascular_invasion"]:
            v = pd.to_numeric(d.get(k), errors="coerce")
            if pd.isna(v):
                rec[k] = np.nan
                coerced_bin += 1
            else:
                rec[k] = int(round(float(v)))

        # categorical
        rec["pT_stage"] = d.get("pT_stage")
        rec["earlier_therapy"] = d.get("earlier_therapy")

        # labels
        bcr = d.get("BCR")
        try: bcr = float(bcr)
        except: pass
        rec["event"] = 1 if bcr in (1, 1.0, "1", "1.0") else 0
        rec["time"]  = pd.to_numeric(d.get("time_to_follow-up/BCR"), errors="coerce")
        rows.append(rec)

    df = pd.DataFrame(rows)
    for leak in ["BCR_PSA","BCR","time_to_follow-up/BCR"]:
        if leak in df.columns:
            df = df.drop(columns=[leak])

    print(f"Non-numeric tokens coerced to missing: {coerced_bin} | df_clin shape: {df.shape}")
    return df

def prep_clin_design(df_clin):
    numeric = ["age_at_prostatectomy","primary_gleason","secondary_gleason","tertiary_gleason",
               "ISUP","pre_operative_PSA","positive_surgical_margins",
               "positive_lymph_nodes","capsular_penetration",
               "invasion_seminal_vesicles","lymphovascular_invasion"]
    categorical = ["pT_stage","earlier_therapy"]

    X_tab = df_clin[numeric + categorical].copy()
    X_tab[categorical] = X_tab[categorical].astype("category")

    
    X_num = X_tab[numeric].to_numpy(dtype=float)
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_num_proc = scaler.fit_transform(imp.fit_transform(X_num))

    
    X_cat = pd.get_dummies(X_tab[categorical], dummy_na=True).to_numpy()

    X_clin = np.hstack([X_num_proc, X_cat]).astype(np.float32)
    return X_clin

def surv_target(df):
    # structured array for sksurv
    y = np.array(list(zip(df["event"].astype(bool), df["time"].astype(float))),
                 dtype=[('event','?'),('time','<f8')])
    return y

def durations_events_from_struct(y):
    # Return float32 for both to avoid mixed types in torchtuples
    durations = y["time"].astype(np.float32)
    events    = y["event"].astype(np.bool_).astype(np.float32)
    return durations, events

def cindex(y, risk):
    return concordance_index_censored(y["event"], y["time"], risk)[0]

def fit_pca(X_tr, X_te, n_comp):
    n_comp = int(min(n_comp, X_tr.shape[0]-5)) if X_tr.shape[0] > 8 else 4
    n_comp = max(n_comp, 4)
    pca = PCA(n_components=n_comp, svd_solver="auto", random_state=RANDOM_STATE)
    return pca.fit_transform(X_tr), pca.transform(X_te), pca, n_comp

def select_alpha_inner_cv(X, y, l1_ratio, alphas_grid, n_splits=3, test_size=0.25, rs=0):
    l1r = max(float(l1_ratio), 1e-6)
    best_alpha, best_score = None, -np.inf
    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=rs)
    for alpha in alphas_grid:
        scores, ok = [], True
        for tr_idx, va_idx in splitter.split(X):
            try:
                mdl = CoxnetSurvivalAnalysis(l1_ratio=l1r, alphas=[alpha], max_iter=200000, tol=1e-6)
                mdl.fit(X[tr_idx], y[tr_idx])
                r = mdl.predict(X[va_idx])
                scores.append(cindex(y[va_idx], r))
            except Exception:
                ok = False
                break
        if ok and scores:
            m = float(np.mean(scores))
            if m > best_score:
                best_score, best_alpha = m, alpha
    if best_alpha is None:
        best_alpha = float(np.max(alphas_grid))
    return best_alpha

def fit_cox(X_tr, y_tr, l1_ratio, alphas_grid):
    alpha = select_alpha_inner_cv(X_tr, y_tr, l1_ratio=l1_ratio, alphas_grid=alphas_grid)
    mdl = CoxnetSurvivalAnalysis(l1_ratio=max(float(l1_ratio),1e-6), alphas=[alpha], max_iter=200000, tol=1e-6)
    mdl.fit(X_tr, y_tr)
    return mdl, alpha

def inner_cv_svm_alpha(X, y, alphas, n_splits=3, test_size=0.25, rs=0):
    best_alpha, best = None, -np.inf
    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=rs)
    for a in alphas:
        vals = []
        for tr_idx, va_idx in splitter.split(X):
            mdl = FastSurvivalSVM(alpha=float(a), max_iter=3000, tol=1e-6, random_state=rs)
            mdl.fit(X[tr_idx], y[tr_idx])
            # risk score (higher = riskier)
            risk = mdl.predict(X[va_idx])
            vals.append(concordance_index_censored(y[va_idx]["event"], y[va_idx]["time"], risk)[0])
        m = float(np.mean(vals))
        if m > best:
            best, best_alpha = m, a
    return best_alpha

# ---------- DeepSurv helpers ----------
def make_mlp(in_features, hidden_layers, dropout):
    layers = []
    prev = in_features
    for h in hidden_layers:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p=dropout)]
        prev = h
    layers += [nn.Linear(prev, 1)]  # CoxPH: outputs log-risk
    return nn.Sequential(*layers)

def train_deepsurv(X_tr, y_tr, X_te, y_te, params=DEEPSURV_PARAMS):
    """
    Train a small DeepSurv (CoxPH NN) with early stopping.
    Ensures X, durations, and events are all *torch tensors* on the same device.
    """
    durations_tr, events_tr = durations_events_from_struct(y_tr)
    durations_te, events_te = durations_events_from_struct(y_te)

    # Standardize inputs (fit on train only)
    scaler = StandardScaler().fit(X_tr)
    Xtr = scaler.transform(X_tr).astype(np.float32)
    Xte = scaler.transform(X_te).astype(np.float32)

    device = "cuda" if (HAS_DEEPSURV and torch.cuda.is_available()) else "cpu"

    
    x_tr = torch.from_numpy(Xtr).to(device)
    x_te = torch.from_numpy(Xte).to(device)
    d_tr = torch.from_numpy(durations_tr).to(device)
    e_tr = torch.from_numpy(events_tr).to(device)
    d_te = torch.from_numpy(durations_te).to(device)
    e_te = torch.from_numpy(events_te).to(device)

    in_features = Xtr.shape[1]
    net = make_mlp(in_features, params["hidden_layers"], params["dropout"]).to(device)

    model = CoxPH(net, tt.optim.Adam, device=device)
    model.optimizer.set_lr(params["lr"])

    batch_size = int(params["batch_size"])
    n_epochs   = int(params["epochs"])
    patience   = int(params["patience"])

    
    y_tr_tuple = (d_tr, e_tr)
    y_te_tuple = (d_te, e_te)

    callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
    model.fit(x_tr, y_tr_tuple,
              batch_size=batch_size, epochs=n_epochs,
              callbacks=callbacks, val_data=(x_te, y_te_tuple),
              num_workers=int(params["num_workers"]), verbose=False,
              shuffle=True, metrics=None)

    with torch.no_grad():
        risk_te = model.predict(x_te).reshape(-1).detach().cpu().numpy().astype(np.float64)

    return model, scaler, risk_te

def eval_block(name, per_fold_scores):
    arr = np.array(per_fold_scores, dtype=float)
    print(f"{name:>12} | C-index per fold: {np.round(arr,3)} | mean={np.nanmean(arr):.3f}±{np.nanstd(arr):.3f}")


# --------------- Main ---------------
def main():
    os.makedirs(OUTD, exist_ok=True)

    X_mri, ids   = load_embeddings()
    folds        = pd.read_csv(FOLDS_CSV, dtype={'patient_id': str})
    df_clin      = build_clinical_table()

    df = ids.merge(df_clin, on="patient_id").merge(folds, on="patient_id")
    assert len(df) == X_mri.shape[0], "Row alignment mismatch"

    y      = surv_target(df)
    X_clin = prep_clin_design(df)

    print(f"Feature blocks prepared. N={len(df)} | MRI={X_mri.shape[1]} | Clin={X_clin.shape[1]}")

    folds_sorted = sorted(df["fold"].unique())
    results = {}
    def add_result(key, val):
        results.setdefault(key, []).append(val)

   
    preds_rows = []

    for k in folds_sorted:
        tr = (df["fold"] != k).values
        te = (df["fold"] == k).values

        X_mri_tr, X_mri_te = X_mri[tr], X_mri[te]
        X_clin_tr, X_clin_te = X_clin[tr], X_clin[te]
        y_tr, y_te = y[tr], y[te]

        
        pca_sets = {}
        for n_comp in PCA_GRID:
            try:
                Xm_tr, Xm_te, _, n_used = fit_pca(X_mri_tr, X_mri_te, n_comp)
                sc_m = StandardScaler().fit(Xm_tr)
                pca_sets[n_used] = (sc_m.transform(Xm_tr).astype(np.float32),
                                    sc_m.transform(Xm_te).astype(np.float32))
            except Exception:
                continue
        if not pca_sets:
            for key in ["RSF_MRI","GBSA_MRI","CGB_MRI","SVM_MRI","COX_MRI","DEEP_MRI",
                        "RSF_CONCAT","GBSA_CONCAT","CGB_CONCAT","SVM_CONCAT","COX_CONCAT","DEEP_CONCAT",
                        "RSF_CLIN","GBSA_CLIN","CGB_CLIN","SVM_CLIN","COX_CLIN","DEEP_CLIN"]:
                add_result(key, np.nan)
            continue

        n_pick = min(pca_sets.keys())
        Xm_tr_s, Xm_te_s = pca_sets[n_pick]

        
        sc_c = StandardScaler().fit(X_clin_tr)
        Xc_tr_s = sc_c.transform(X_clin_tr).astype(np.float32)
        Xc_te_s = sc_c.transform(X_clin_te).astype(np.float32)

        # ===== MRI-only =====
        if RUN_MRI_ONLY:
            if USE_RSF:
                rsf = RandomSurvivalForest(**RSF_PARAMS).fit(Xm_tr_s, y_tr)
                risk = -rsf.predict(Xm_te_s)
                add_result("RSF_MRI", cindex(y_te, risk))
            if USE_GBSA:
                gb = GradientBoostingSurvivalAnalysis(loss="coxph", **GBSA_PARAMS).fit(Xm_tr_s, y_tr)
                risk = gb.predict(Xm_te_s)
                add_result("GBSA_MRI", cindex(y_te, risk))
            if USE_CGB and HAS_CGB:
                cgb = ComponentwiseGradientBoostingSurvivalAnalysis(**CGB_PARAMS).fit(Xm_tr_s, y_tr)
                risk = cgb.predict(Xm_te_s)
                add_result("CGB_MRI", cindex(y_te, risk))
            if USE_SVM and HAS_SVM:
                best_a = inner_cv_svm_alpha(Xm_tr_s, y_tr, alphas=SVM_ALPHAS,
                                            n_splits=INNER_SPLITS, test_size=INNER_TEST, rs=RANDOM_STATE)
                svm = FastSurvivalSVM(alpha=float(best_a), max_iter=3000, tol=1e-6, random_state=RANDOM_STATE).fit(Xm_tr_s, y_tr)
                risk = svm.predict(Xm_te_s)
                add_result("SVM_MRI", cindex(y_te, risk))
            if USE_COX:
                mdl, _ = fit_cox(Xm_tr_s, y_tr, l1_ratio=L1R_MRI, alphas_grid=ALPHAS_RIDGE)
                risk = mdl.predict(Xm_te_s)
                add_result("COX_MRI", cindex(y_te, risk))
            if USE_DEEPSURV and HAS_DEEPSURV:
                try:
                    _, _, risk = train_deepsurv(Xm_tr_s, y_tr, Xm_te_s, y_te)
                    add_result("DEEP_MRI", cindex(y_te, risk))
                except Exception as e:
                    print(f"[DeepSurv MRI] skipped: {e}")
                    add_result("DEEP_MRI", np.nan)

        # ===== Clin-only =====
        if RUN_CLIN_ONLY:
            if USE_RSF:
                rsf = RandomSurvivalForest(**RSF_PARAMS).fit(X_clin_tr, y_tr)
                risk = -rsf.predict(X_clin_te)
                add_result("RSF_CLIN", cindex(y_te, risk))
            if USE_GBSA:
                gb = GradientBoostingSurvivalAnalysis(loss="coxph", **GBSA_PARAMS).fit(Xc_tr_s, y_tr)
                risk = gb.predict(Xc_te_s)
                add_result("GBSA_CLIN", cindex(y_te, risk))
            if USE_CGB and HAS_CGB:
                cgb = ComponentwiseGradientBoostingSurvivalAnalysis(**CGB_PARAMS).fit(Xc_tr_s, y_tr)
                risk = cgb.predict(Xc_te_s)
                add_result("CGB_CLIN", cindex(y_te, risk))
            if USE_SVM and HAS_SVM:
                best_a = inner_cv_svm_alpha(Xc_tr_s, y_tr, alphas=SVM_ALPHAS,
                                            n_splits=INNER_SPLITS, test_size=INNER_TEST, rs=RANDOM_STATE)
                svm = FastSurvivalSVM(alpha=float(best_a), max_iter=3000, tol=1e-6, random_state=RANDOM_STATE).fit(Xc_tr_s, y_tr)
                risk = svm.predict(Xc_te_s)
                add_result("SVM_CLIN", cindex(y_te, risk))
            if USE_COX:
                # ---- COX_CLIN: also store case-wise predictions as "best classical model" ----
                mdl, _ = fit_cox(Xc_tr_s, y_tr, l1_ratio=L1R_CLIN, alphas_grid=ALPHAS_RIDGE)
                risk = mdl.predict(Xc_te_s)
                add_result("COX_CLIN", cindex(y_te, risk))

                # Store predictions for each test patient for this fold
                ids_te = df.loc[te, "patient_id"].values
                for i, pid in enumerate(ids_te):
                    preds_rows.append(
                        {
                            "patient_id": pid,
                            "fold": int(k),
                            "event": int(y_te[i]["event"]),
                            "time": float(y_te[i]["time"]),
                            "risk_cox_clin": float(risk[i]),
                        }
                    )

            if USE_DEEPSURV and HAS_DEEPSURV:
                try:
                    _, _, risk = train_deepsurv(Xc_tr_s, y_tr, Xc_te_s, y_te)
                    add_result("DEEP_CLIN", cindex(y_te, risk))
                except Exception as e:
                    print(f"[DeepSurv Clin] skipped: {e}")
                    add_result("DEEP_CLIN", np.nan)

        # ===== MRI + Clin =====
        if RUN_MRI_PLUS_CLIN:
            Xtr_cat = np.hstack([Xm_tr_s, X_clin_tr]).astype(np.float32)
            Xte_cat = np.hstack([Xm_te_s, X_clin_te]).astype(np.float32)

            sc_cat = StandardScaler().fit(Xtr_cat)
            Xtr_cat_s = sc_cat.transform(Xtr_cat).astype(np.float32)
            Xte_cat_s = sc_cat.transform(Xte_cat).astype(np.float32)

            if USE_RSF:
                rsf = RandomSurvivalForest(**RSF_PARAMS).fit(Xtr_cat, y_tr)
                risk = -rsf.predict(Xte_cat)
                add_result("RSF_CONCAT", cindex(y_te, risk))
            if USE_GBSA:
                gb = GradientBoostingSurvivalAnalysis(loss="coxph", **GBSA_PARAMS).fit(Xtr_cat_s, y_tr)
                risk = gb.predict(Xte_cat_s)
                add_result("GBSA_CONCAT", cindex(y_te, risk))
            if USE_CGB and HAS_CGB:
                cgb = ComponentwiseGradientBoostingSurvivalAnalysis(**CGB_PARAMS).fit(Xtr_cat_s, y_tr)
                risk = cgb.predict(Xte_cat_s)
                add_result("CGB_CONCAT", cindex(y_te, risk))
            if USE_SVM and HAS_SVM:
                best_a = inner_cv_svm_alpha(Xtr_cat_s, y_tr, alphas=SVM_ALPHAS,
                                            n_splits=INNER_SPLITS, test_size=INNER_TEST, rs=RANDOM_STATE)
                svm = FastSurvivalSVM(alpha=float(best_a), max_iter=3000, tol=1e-6, random_state=RANDOM_STATE).fit(Xtr_cat_s, y_tr)
                risk = svm.predict(Xte_cat_s)
                add_result("SVM_CONCAT", cindex(y_te, risk))
            if USE_COX:
                mdl, _ = fit_cox(Xtr_cat_s, y_tr, l1_ratio=L1R_CONCAT, alphas_grid=ALPHAS_ELNET)
                risk = mdl.predict(Xte_cat_s)
                add_result("COX_CONCAT", cindex(y_te, risk))
            if USE_DEEPSURV and HAS_DEEPSURV:
                try:
                    _, _, risk = train_deepsurv(Xtr_cat_s, y_tr, Xte_cat_s, y_te)
                    add_result("DEEP_CONCAT", cindex(y_te, risk))
                except Exception as e:
                    print(f"[DeepSurv Concat] skipped: {e}")
                    add_result("DEEP_CONCAT", np.nan)

    # ---- Summaries----
    for key, vals in sorted(results.items()):
        arr = np.array(vals, dtype=float)
        print(f"{key:>12} | C-index per fold: {np.round(arr,3)} | mean={np.nanmean(arr):.3f}±{np.nanstd(arr):.3f}")

    # ---- Save per-fold metrics to CSV ----
    if results:
       
        metrics_df = pd.DataFrame(
            {key: np.array(vals, dtype=float) for key, vals in sorted(results.items())}
        )
        metrics_df.insert(0, "fold", folds_sorted)
        metrics_path = os.path.join(OUTD, "classical_cindex_per_fold.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"[info] Saved per-fold classical metrics to {metrics_path}")

    # ---- Save COX_CLIN case-wise predictions to CSV ----
    if preds_rows:
        preds_df = pd.DataFrame(preds_rows)
        preds_path = os.path.join(OUTD, "cox_clin_casewise_predictions.csv")
        preds_df.to_csv(preds_path, index=False)
        print(f"[info] Saved COX_CLIN case-wise predictions to {preds_path}")


if __name__ == "__main__":
    if not HAS_DEEPSURV and USE_DEEPSURV:
        print("[Info] DeepSurv requested but pycox/torchtuples not found. Install with: pip install pycox torchtuples")
    main()
