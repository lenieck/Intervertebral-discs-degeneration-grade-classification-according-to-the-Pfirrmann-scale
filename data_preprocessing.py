import os
import zipfile
import pandas as pd
import numpy as np
import SimpleITK as sitk
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

BASE_PATH = 'SPIDER'
IMAGES_ZIP = os.path.join(BASE_PATH, 'images.zip')
MASKS_ZIP = os.path.join(BASE_PATH, 'masks.zip')

IMAGES_CLEANED = os.path.join(BASE_PATH, 'images_cleaned')
MASKS_CLEANED = os.path.join(BASE_PATH, 'masks_cleaned')

OUTPUT_ROOT = os.path.join(BASE_PATH, 'classification_dataset')
CSV_GRADINGS = os.path.join(BASE_PATH, 'radiological_gradings.csv')

TARGET_SPACING = (1.0, 1.0, 1.0)
CLASSIFICATION_SIZE = (128, 128)  
VAL_SIZE = 0.2

def setup_directories():
    for split in ['train', 'val']:
        for grade in range(1, 6):
            os.makedirs(os.path.join(OUTPUT_ROOT, split, f"grade_{grade}"), exist_ok=True)
    os.makedirs(IMAGES_CLEANED, exist_ok=True)
    os.makedirs(MASKS_CLEANED, exist_ok=True)


def extract_only_t2(zip_path, target_folder):
    """Extracts only *_t2.mha files from zip."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        for member in z.namelist():
            if member.endswith('_t2.mha'):
                filename = os.path.basename(member)
                if filename:
                    with z.open(member) as source, open(os.path.join(target_folder, filename), "wb") as target:
                        target.write(source.read())

def resample_image(itk_image, is_mask=False):
    """Standardizes pixel spacing to 1.0mm isotropic."""
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    new_size = [int(round(original_size[i] * (original_spacing[i] / TARGET_SPACING[i]))) for i in range(3)]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(TARGET_SPACING)
    resample.SetSize(new_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return resample.Execute(itk_image)

def pad_to_fixed_size(image_array, target_size):
    """Pads image to target size. Resizes down if image exceeds target size."""
    h, w = image_array.shape
    th, tw = target_size
    
    if h > th or w > tw:
        img_temp = Image.fromarray(image_array)
        img_temp.thumbnail(target_size, Image.Resampling.LANCZOS)
        image_array = np.array(img_temp)
        h, w = image_array.shape

    pad_h = max(0, (th - h) // 2)
    pad_w = max(0, (tw - w) // 2)
    padded = np.pad(image_array, ((pad_h, th - h - pad_h), (pad_w, tw - w - pad_w)), 
                    mode='constant', constant_values=0)
    return padded


setup_directories()
extract_only_t2(IMAGES_ZIP, IMAGES_CLEANED)
extract_only_t2(MASKS_ZIP, MASKS_CLEANED)

df_gradings = pd.read_csv(CSV_GRADINGS)
image_paths = glob(os.path.join(IMAGES_CLEANED, "*.mha"))
patient_ids = list(set([os.path.basename(p).split('_')[0] for p in image_paths]))

train_ids, val_ids = train_test_split(patient_ids, test_size=VAL_SIZE, random_state=42)
print(f"Data Split: {len(train_ids)} patients for Training, {len(val_ids)} for Validation.")

for img_path in image_paths:
    file_name = os.path.basename(img_path)
    case_id = file_name.replace('.mha', '')
    patient_id = case_id.split('_')[0]
    split = 'train' if patient_id in train_ids else 'val'
    
    mask_path = os.path.join(MASKS_CLEANED, file_name)
    if not os.path.exists(mask_path):
        continue

    itk_img = resample_image(sitk.DICOMOrient(sitk.ReadImage(img_path), 'LPS'), is_mask=False)
    itk_msk = resample_image(sitk.DICOMOrient(sitk.ReadImage(mask_path), 'LPS'), is_mask=True)

    img_array = sitk.GetArrayFromImage(itk_img)
    msk_array = sitk.GetArrayFromImage(itk_msk)
    
    mid_x = img_array.shape[2] // 2
    for offset in [-1, 0, 1]:
        slice_idx = mid_x + offset
        if slice_idx < 0 or slice_idx >= img_array.shape[2]:
            continue
        
        img_slice = np.flipud(img_array[:, :, slice_idx])
        msk_slice = np.flipud(msk_array[:, :, slice_idx])

        p1, p99 = np.percentile(img_slice, [1, 99])
        img_norm = ((np.clip(img_slice, p1, p99) - p1) / (p99 - p1) * 255).astype(np.uint8)

        for mask_val, ivd_idx in {201: 1, 202: 2, 203: 3, 204: 4, 205: 5}.items():
            match = df_gradings[(df_gradings['Patient'] == int(patient_id)) & (df_gradings['IVD label'] == ivd_idx)]
            if match.empty:
                continue
            
            grade = int(match['Pfirrman grade'].values[0])
            coords = np.column_stack(np.where(msk_slice == mask_val))
            
            if coords.size > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                crop = img_norm[max(0, y_min-10):y_max+10, max(0, x_min-10):x_max+10]
                padded_crop = pad_to_fixed_size(crop, CLASSIFICATION_SIZE)
                
                save_dir = os.path.join(OUTPUT_ROOT, split, f"grade_{grade}")
                filename = f"p{patient_id}_s{offset+1}_d{ivd_idx}.png"
                Image.fromarray(padded_crop).save(os.path.join(save_dir, filename))