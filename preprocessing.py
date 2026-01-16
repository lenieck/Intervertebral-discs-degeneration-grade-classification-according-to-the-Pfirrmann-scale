import os
import shutil
import SimpleITK as sitk
import numpy as np
import cv2
import pandas as pd

SOURCE_DIR = "SPIDER"
TARGET_DIR = "SPIDER_cleaned"
OUTPUT_DIR = 'SPIDER_training_margin5_patient_split'
GRADES = [1, 2, 3, 4, 5]
MARGIN = 5

def filter_t2(src_dir=SOURCE_DIR, target_dir=TARGET_DIR):
    for subfolder in ["images", "masks"]:
        src_path = os.path.join(src_dir, subfolder)
        dst_path = os.path.join(target_dir, subfolder)
        os.makedirs(dst_path, exist_ok=True)

        for f in os.listdir(src_path):
            if f.endswith("_t2.mha"):
                shutil.copy2(os.path.join(src_path, f), os.path.join(dst_path, f))

def normalize_to_png(img_slice):
    img_min = np.min(img_slice)
    img_max = np.max(img_slice)
    if img_max == img_min:
        return np.zeros_like(img_slice, dtype=np.uint8)
    img_norm = 255.0 * (img_slice - img_min) / (img_max - img_min)
    return img_norm.astype(np.uint8)

def tight_crop(img_slice, mask_slice, disc_label, margin=MARGIN):
    y_idx, x_idx = np.where(mask_slice == disc_label)
    if len(y_idx) == 0 or len(x_idx) == 0:
        return None
    y_min, y_max = np.min(y_idx), np.max(y_idx)
    x_min, x_max = np.min(x_idx), np.max(x_idx)
    y_min = max(y_min - margin, 0)
    y_max = min(y_max + margin, img_slice.shape[0] - 1)
    x_min = max(x_min - margin, 0)
    x_max = min(x_max + margin, img_slice.shape[1] - 1)
    return img_slice[y_min:y_max+1, x_min:x_max+1]

def preprocess_dataset(base_dir=TARGET_DIR, output_dir=OUTPUT_DIR, grades=GRADES, margin=MARGIN):
    img_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")

    for split in ['train', 'val']:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
        for grade in grades:
            os.makedirs(os.path.join(split_dir, f'grade{grade}'))

    csv_path = os.path.join(base_dir, 'radiological_gradings.csv')
    df = pd.read_csv(csv_path)
    df['DiskMaskLabel'] = df['IVD label'] + 200
    df = df[df['DiskMaskLabel'].isin([201,202,203,204,205])]

    overview_path = os.path.join(base_dir, 'overview.csv')
    overview = pd.read_csv(overview_path)
    overview['Patient'] = overview['new_file_name'].str.split('_').str[0].astype(int)
    patient_to_split = overview[['Patient', 'subset']].drop_duplicates().set_index('Patient')['subset'].to_dict()

    files = sorted([f for f in os.listdir(img_dir) if f.endswith('.mha')])

    for filename in files:
        patient_id = int(filename.split("_")[0])
        subset = patient_to_split.get(patient_id)
        if subset is None:
            print(f"Patient {patient_id} not included in split.")
            continue
        split_type = 'train' if subset == 'training' else 'val'

        img_path = os.path.join(img_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        if not os.path.exists(mask_path):
            print(f"Mask missing: {filename}")
            continue

        try:
            sitk_img = sitk.ReadImage(img_path)
            sitk_mask = sitk.ReadImage(mask_path)

            orient_filter = sitk.DICOMOrientImageFilter()
            orient_filter.SetDesiredCoordinateOrientation("LPS")
            sitk_img = orient_filter.Execute(sitk_img)
            sitk_mask = orient_filter.Execute(sitk_mask)

            arr_img = sitk.GetArrayFromImage(sitk_img)
            arr_mask = sitk.GetArrayFromImage(sitk_mask)

            for disc_label in range(201, 206):
                df_row = df[(df['Patient']==patient_id) & (df['DiskMaskLabel']==disc_label)]
                if df_row.empty:
                    continue
                pf_grade = int(df_row['Pfirrman grade'].values[0])

                mask_slices = np.any(arr_mask == disc_label, axis=1)
                x_indices = np.where(mask_slices.any(axis=0))[0]
                if len(x_indices) == 0:
                    continue
                x_center = int(np.mean(x_indices))

                offsets = [-1,0,1] if pf_grade in [2,3,4] else [-2,-1,0,1,2]
                valid_offsets = [o for o in offsets if 0 <= x_center + o < arr_img.shape[2]]

                for o in valid_offsets:
                    x_idx = x_center + o
                    img_slice = np.flipud(arr_img[:, :, x_idx])
                    mask_slice = np.flipud(arr_mask[:, :, x_idx])

                    cropped = tight_crop(img_slice, mask_slice, disc_label, margin)
                    if cropped is None:
                        continue

                    cropped = normalize_to_png(cropped)
                    out_dir = os.path.join(output_dir, split_type, f'grade{pf_grade}')
                    base_name = f"patient{patient_id}_disk{disc_label}_slice{o}_pf{pf_grade}.png"
                    out_path = os.path.join(out_dir, base_name)
                    cv2.imwrite(out_path, cropped)

        except Exception as e:
            print(f"Error {filename}: {e}")

if __name__ == "__main__":
    filter_t2()
    preprocess_dataset()