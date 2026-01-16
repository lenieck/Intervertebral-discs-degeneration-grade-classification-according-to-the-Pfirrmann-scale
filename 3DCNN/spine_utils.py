"""
Shared utilities for SpineNet disc extraction and Pfirrmann grading.
"""
import pathlib
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
from sklearn.model_selection import train_test_split

# Paths
PROJECT_PATH = pathlib.Path("/home/mpuscian/Desktop/repozytoria/MINI_projects/Intervertebral-discs-degeneration-grade-classification-according-to-the-Pfirrmann-scale/")
DATA_PATH = PROJECT_PATH / "data/SPIDER_kaggle/"
IMAGES_PATH = DATA_PATH / "images/images/"
MASKS_PATH = DATA_PATH / "masks/masks/"

# Disc label mapping: IVD label -> mask label
DISC_MAP = {1: 201, 2: 202, 3: 203, 4: 204, 5: 205, 6: 206, 7: 207}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ Model Definition ============

def conv3x3(inp, out, stride=1):
    return nn.Conv3d(inp, out, 3, stride=(1, stride, stride), padding=1, bias=False)

def conv1x1(inp, out, stride=1):
    return nn.Conv3d(inp, out, 1, stride=(1, stride, stride), bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inp, out, stride=1, downsample=None):
        super().__init__()
        self.conv1, self.bn1 = conv3x3(inp, out, stride), nn.BatchNorm3d(out)
        self.conv2, self.bn2 = conv3x3(out, out), nn.BatchNorm3d(out)
        self.relu, self.downsample = nn.ReLU(inplace=True), downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x) if self.downsample else x
        return self.relu(out)

class PfirrmannModel(nn.Module):
    """3D ResNet for Pfirrmann grade classification."""
    def __init__(self, layers=[2,2,2,2], num_classes=5):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, (3,7,7), stride=(1,2,2), padding=(1,3,3))
        self.bn1, self.relu = nn.BatchNorm3d(64), nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=1)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes, stride), nn.BatchNorm3d(planes))
        layers = [BasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [BasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def load_model(model_path=None):
    """Load trained Pfirrmann model."""
    if model_path is None:
        model_path = PROJECT_PATH / "SpineNet/pfirrmann_model.pth"
    model = PfirrmannModel().to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


# ============ Data Loading ============

def get_t2_file_pairs():
    """Get all T2 image and mask file pairs."""
    files = []
    for f in sorted(IMAGES_PATH.glob("*_t2.mha")):
        pid = int(f.stem.replace("_t2", ""))
        msk = MASKS_PATH / f.name
        if msk.exists():
            files.append({"pid": pid, "img": f, "msk": msk})
    return files

def split_data_by_patient(file_pairs, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split files by patient into train/val/test."""
    pids = list(set(f["pid"] for f in file_pairs))
    train_pids, temp_pids = train_test_split(pids, train_size=train_ratio, random_state=seed)
    val_adj = val_ratio / (val_ratio + test_ratio)
    val_pids, test_pids = train_test_split(temp_pids, train_size=val_adj, random_state=seed)

    train = [f for f in file_pairs if f["pid"] in set(train_pids)]
    val = [f for f in file_pairs if f["pid"] in set(val_pids)]
    test = [f for f in file_pairs if f["pid"] in set(test_pids)]
    return train, val, test


# ============ Disc Extraction ============

def resize_volume(vol, shape=(224, 224, 16)):
    """Resize volume to target shape with 95th percentile normalization."""
    t = torch.einsum("ijk->kij", torch.tensor(vol)).unsqueeze(1).double()
    t = F.interpolate(t, size=shape[:2], mode='bicubic', align_corners=False).squeeze(1)
    t = torch.einsum("kij->ijk", t)

    out = torch.zeros(shape)
    ratio = t.shape[-1] / shape[-1]
    for i in range(shape[-1]):
        idx = min(int(i * ratio), t.shape[-1] - 1)
        out[:, :, i] = t[:, :, idx]

    p95, mn = np.percentile(out, 95), out.min()
    return ((out - mn) / (p95 + 1e-6)).numpy()


def extract_disc(scan_sitk, mask_sitk, disc_label, output_shape=(224, 224, 16), extent=1.0):
    """Extract a single disc volume from scan using mask."""
    scan = sitk.GetArrayFromImage(scan_sitk).astype(np.float32)
    mask = sitk.GetArrayFromImage(mask_sitk)

    disc_mask = (mask == disc_label).astype(np.uint8)
    if disc_mask.sum() == 0:
        return None

    # Project to 2D and find contour
    mask_2d = disc_mask.max(axis=2).astype(np.uint8)
    contours, _ = cv2.findContours(mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Get oriented bounding box
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    (cx, cy), (rw, rh), ang = rect
    rot_ang = ang if rw >= rh else ang + 90

    # Rotate scan
    M = cv2.getRotationMatrix2D((cx, cy), rot_ang, 1)
    h, w, d = scan.shape
    rot_scan = np.stack([cv2.warpAffine(scan[:, :, i], M, (w, h)) for i in range(d)], axis=2)

    # Crop around disc
    edge = max(rw, rh) * (1 + extent)
    x1, x2 = int(cx - edge/2), int(cx + edge/2)
    y1, y2 = int(cy - edge/2), int(cy + edge/2)
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    vol = rot_scan[y1:y2, x1:x2, :]
    if vol.size == 0:
        return None

    return resize_volume(vol, output_shape)


def extract_all_discs(scan_sitk, mask_sitk, output_shape=(224, 224, 16), extent=1.0):
    """Extract all disc volumes from a scan."""
    mask = sitk.GetArrayFromImage(mask_sitk)
    disc_labels = [l for l in np.unique(mask) if l >= 200]

    volumes = {}
    for label in disc_labels:
        vol = extract_disc(scan_sitk, mask_sitk, int(label), output_shape, extent)
        if vol is not None:
            volumes[int(label)] = vol
    return volumes


# ============ Inference ============

def predict(model, volume):
    """Get prediction for a disc volume."""
    t = torch.from_numpy(volume).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(t)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred = out.argmax(1).item() + 1  # 1-indexed grade
    return pred, probs
