import os, json, glob
import numpy as np
import SimpleITK as sitk
import pandas as pd

ROOT = r"D:\OneDrive - Queen's University\Multimodal-Quiz\Multimodal-Quiz"
IN_CLIN = os.path.join(ROOT, "clinical_data")
IN_MPMRI = os.path.join(ROOT, "radiology", "mpMRI")
IN_MASKS = os.path.join(ROOT, "radiology", "prostate_mask_t2w")

OUT_NII = os.path.join(ROOT, "outputs", "nii_resampled")
os.makedirs(OUT_NII, exist_ok=True)

def read_mha(path):
    return sitk.ReadImage(path)

def to_nii(img, path):
    sitk.WriteImage(img, path)

def resample_to_ref(moving, ref, interp=sitk.sitkLinear, default=0.0, out_pixel_type=None):
    """
    Resample 'moving' image onto 'ref' image geometry using the reference-image overload.
    """
    if out_pixel_type is None:
        out_pixel_type = moving.GetPixelID()  
    return sitk.Resample(
        moving,             
        ref,                
        sitk.Transform(),   
        interp,             
        default,            
        out_pixel_type     
    )


rows = []
# Expect patient folders named like "1003", and each has adc.mha, hbv.mha, t2w.mha 
for pid_dir in sorted(os.listdir(IN_MPMRI)):
    pid = pid_dir.strip()
    pdir = os.path.join(IN_MPMRI, pid)
    if not os.path.isdir(pdir): 
        continue

    # Find files 
    def find_one(patterns):
        for p in patterns:
            g = glob.glob(os.path.join(pdir, p))
            if g: return g[0]
        return None

    f_t2 = find_one(["*t2*.mha", "*T2*.mha"])
    f_adc = find_one(["*adc*.mha", "*ADC*.mha"])
    f_hbv = find_one(["*hbv*.mha", "*HBV*.mha", "*hb*.mha"])

    f_mask = os.path.join(IN_MASKS, f"{pid}_0001_mask.mha")
    if not (f_t2 and f_adc and f_hbv and os.path.exists(f_mask)):
        print(f"[WARN] Missing modality for patient {pid}")
        continue

    # Read images
    img_t2  = read_mha(f_t2)
    img_adc = read_mha(f_adc)
    img_hbv = read_mha(f_hbv)
    msk_t2  = read_mha(f_mask)

       
    
    
    # Images
    img_adc_r = resample_to_ref(img_adc, img_t2, interp=sitk.sitkLinear)
    img_hbv_r = resample_to_ref(img_hbv, img_t2, interp=sitk.sitkLinear)
    
    # Mask: nearest-neighbor + force binary UINT8
    msk_t2_r = resample_to_ref(
        msk_t2, img_t2,
        interp=sitk.sitkNearestNeighbor,
        default=0.0,
        out_pixel_type=sitk.sitkUInt8
    )
    # Binarize: >0 → 1, else 0
    msk_t2_r = sitk.Cast(msk_t2_r > 0, sitk.sitkUInt8)

    
    
    
    # Sanity checks
    # assert img_adc_r.GetSize()       == img_t2.GetSize()
    # assert img_adc_r.GetSpacing()    == img_t2.GetSpacing()
    # assert img_adc_r.GetOrigin()     == img_t2.GetOrigin()
    # assert img_adc_r.GetDirection()  == img_t2.GetDirection()
    
    # assert msk_t2_r.GetSize()      == img_t2.GetSize()
    # assert msk_t2_r.GetSpacing()   == img_t2.GetSpacing()
    # assert msk_t2_r.GetOrigin()    == img_t2.GetOrigin()
    # assert msk_t2_r.GetDirection() == img_t2.GetDirection()

    # # Inspect values
    # stats = sitk.StatisticsImageFilter()
    # stats.Execute(msk_t2_r)
    # print("Mask min/max:", stats.GetMinimum(), stats.GetMaximum())  # should be 0/1




    

    # Write NIfTIs
    out_pid = os.path.join(OUT_NII, pid)
    os.makedirs(out_pid, exist_ok=True)
    out_t2  = os.path.join(out_pid, "t2w.nii.gz")
    out_adc = os.path.join(out_pid, "adc.nii.gz")
    out_hbv = os.path.join(out_pid, "hbv.nii.gz")
    out_msk = os.path.join(out_pid, "prostate_mask_t2w.nii.gz")

    to_nii(img_t2,  out_t2)
    to_nii(img_adc_r, out_adc)
    to_nii(img_hbv_r, out_hbv)
    to_nii(msk_t2_r,  out_msk)

    rows.append({"patient_id": pid, "t2w": out_t2, "adc": out_adc, "hbv": out_hbv, "mask_t2w": out_msk})
    
    

df = pd.DataFrame(rows).sort_values("patient_id")
df.to_csv(os.path.join(ROOT, "outputs", "paths_index.csv"), index=False)
print("Saved:", os.path.join(ROOT, "outputs", "paths_index.csv"))
