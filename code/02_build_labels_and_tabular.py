import os, json, glob
import pandas as pd
import numpy as np

ROOT = r"D:\OneDrive - Queen's University\Multimodal-Quiz\Multimodal-Quiz"
IN_CLIN = os.path.join(ROOT, "clinical_data")
SPLIT_CSV = os.path.join(ROOT, "data_split_5fold.csv")

paths_index = pd.read_csv(os.path.join(ROOT, "outputs", "paths_index.csv"), dtype={"patient_id": str})
splits = pd.read_csv(SPLIT_CSV, sep=None, engine="python", dtype=str)

rows = []
for jf in sorted(glob.glob(os.path.join(IN_CLIN, "*.json"))):
    pid = os.path.splitext(os.path.basename(jf))[0]  # "1003"
    with open(jf, "r") as f:
        data = json.load(f)

    def to01(x):
        if x is None: return np.nan
        s = str(x).strip().lower()
        if s in ("1","1.0","true","yes"): return 1
        if s in ("0","0.0","false","no","none","nan"): return 0
        try:
            return int(float(s))
        except:
            return np.nan

    y_event = to01(data.get("BCR"))
    y_time  = float(data.get("time_to_follow-up/BCR"))

    # clinical features allowed 
    row = {
        "patient_id": pid,
        "age_at_prostatectomy": float(data.get("age_at_prostatectomy", np.nan)),
        "primary_gleason": float(data.get("primary_gleason", np.nan)),
        "secondary_gleason": float(data.get("secondary_gleason", np.nan)),
        "tertiary_gleason": float(data.get("tertiary_gleason", np.nan)),
        "ISUP": float(data.get("ISUP", np.nan)),
        "pre_operative_PSA": float(data.get("pre_operative_PSA", np.nan)),
        "pT_stage": str(data.get("pT_stage", "")),
        "positive_lymph_nodes": to01(data.get("positive_lymph_nodes")),
        "capsular_penetration": to01(data.get("capsular_penetration")),
        "positive_surgical_margins": to01(data.get("positive_surgical_margins")),
        "invasion_seminal_vesicles": to01(data.get("invasion_seminal_vesicles")),
        "lymphovascular_invasion": to01(data.get("lymphovascular_invasion")),
        "earlier_therapy": str(data.get("earlier_therapy", "none")).strip().lower(),
        # labels
        "y_event": int(y_event) if not np.isnan(y_event) else np.nan,
        "y_time": y_time
    }
    # DO NOT include BCR_PSA as a feature (leakage)
    rows.append(row)

df = pd.DataFrame(rows)

# Encode simple categoricals
df["pT_stage_num"] = df["pT_stage"].map({
    "2":2,"2a":2.1,"2b":2.2,"2c":2.3,
    "3":3,"3a":3.1,"3b":3.2,
    "4":4,"4a":4.1,"4b":4.2
}).fillna(np.nan)

df["earlier_therapy_none"] = (df["earlier_therapy"]=="none").astype(int)

# Merge with paths and folds
df = df.merge(paths_index, on="patient_id", how="inner")
df = df.merge(splits.astype({"patient_id":str}), on="patient_id", how="left")

out_csv = os.path.join(ROOT, "outputs", "clinical_tabular_labels.csv")
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)
