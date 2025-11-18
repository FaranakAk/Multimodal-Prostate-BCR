# Multimodal Prognosis of Biochemical Recurrence in Prostate Cancer

This repository contains my solution to the **Multimodal Coding Test** on prostate cancer prognosis from mpMRI and clinical data.

The goal is to predict **time to biochemical recurrence (BCR)** using:
- Multiparametric MRI (T2w, ADC, HBV)
- Clinical variables
- 5-fold cross-validation with **censored C-index** as the evaluation metric.

---

## 1. Environment

Tested with:

- Python 3.10
- PyTorch >= 2.0
- NumPy, SciPy, pandas, scikit-learn
- `scikit-survival` (for RSF, GBSA, etc.)
- `lifelines` or `scikit-survival`-compatible Cox models
- `wandb` for experiment logging
- `nibabel`, `SimpleITK` for medical image IO

## 2. Data layout
The code expects the following directory layout under ROOT (configured at the top of each script):

```text
ROOT/
├─ clinical_data/                  # *.json clinical files
├─ radiology/
│   ├─ mpMRI/                      # patient folders with *.mha MRI volumes
│   └─ prostate_mask_t2w/          # prostate mask *.mha
├─ pretrained_weights/
│   ├─ t2_extractor.pth
│   ├─ adc_extractor.pth
│   └─ dwi_extractor.pth
├─ data_split_5fold.csv            # provided split file
└─ outputs/                        # will be created by scripts
```





## 3. Pipeline
### Step 1 – Convert and resample MRI to NIfTI
python 01_convert_resample_to_nii.py

This script:
- Converts .mha volumes and masks to .nii.gz,
- Resamples ADC and HBV to T2 geometry,
- Writes a paths_index.csv file in outputs/.

### Step 2 – Build clinical table and labels
python 02_build_labels_and_tabular.py

This script:
- Parses clinical JSON files into a tabular dataframe,
- Encodes clinical features and survival labels,
- Optionally writes a summary CSV.

### Step 3a – Global MRI embeddings (classical models)
python 03_extract_mri_embeddings.py

This script:

- Loads NIfTI MRI volumes and prostate masks,
- Extracts slice-level features for T2, ADC, HBV using pretrained CNNs,
- Aggregates them (median) into a 1728-d MRI embedding per patient,
- Saves:
outputs/X_mri_cnn.npy
outputs/X_mri_ids.csv.

### Step 3b – Slice-level MRI bags (ABMIL model)
python 033_extract_mri_bags.py

This script:
- Builds modality-aligned slices inside the prostate,
- Normalizes and crops patches,
- Extracts per-slice CNN features and concatenates them into 1728-d vectors,
- Saves patient bags in outputs/mri_bags/*.npz and an index in
outputs/mri_bags_index.csv.

### Step 4 – Classical survival baselines
python 04_train_survival_5fold.py

This script:
- Loads MRI embeddings and clinical features,
- Trains multiple survival models (Cox, RSF, GBSA, SVM, DeepSurv) on:
MRI-only
clinical-only
MRI+clinical (concatenated)

- Uses the provided 5-fold split data_split_5fold.csv for cross-validation,

- Prints C-index per fold and mean ± SD for each model,

Saves:
outputs/classical_cindex_per_fold.csv
outputs/cox_clin_casewise_predictions.csv.


### Step 5 – Multimodal ABMIL model
- Full multimodal model (attention + clinical)
python 044_train_abmil_survival.py  # with EXPERIMENT_TAG="full", ABLATE_NO_CLIN=False, ABLATE_MEAN_POOL=False

- MRI-only ABMIL (no clinical)
set EXPERIMENT_TAG="noclin", ABLATE_NO_CLIN=True, ABLATE_MEAN_POOL=False, then:
python 044_train_abmil_survival.py

- Mean pooling instead of attention
set EXPERIMENT_TAG="meanpool", ABLATE_NO_CLIN=False, ABLATE_MEAN_POOL=True, then:
python 044_train_abmil_survival.py


Each run:
- Logs train/val loss and test C-index per fold to wandb,
- Saves attention weights per test patient in outputs/attn_<tag>/fold_k/*.csv,
- Saves fold-wise checkpoints in outputs/checkpoints/abmil_<tag>_fold{k}.pth,
- Saves case-wise predictions to:
outputs/abmil_casewise_predictions_<tag>.csv.
