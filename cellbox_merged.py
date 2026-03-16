print("DEBUG: IMPORT - 1")
import cv2
import torch
import torch.nn as nn
import numpy as np

print("DEBUG: IMPORT - 2")
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor, ViTModel, AutoModel

print("DEBUG: IMPORT - 3")
from PIL import Image
import os
from cellpose import models, utils
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

print("DEBUG: IMPORT - 4")
from collections import defaultdict, Counter, deque
from cellpose import plot
from skimage.metrics import structural_similarity as ssim
from skimage.measure import label, regionprops
import argparse
import os
import pickle
import torch.nn.functional as F

print("DEBUG: IMPORT - 5")

# =============================================================================
# ─── CONSTANTS ────────────────────────────────────────────────────────────────
# =============================================================================

# Note (kaiyu): in cv2, color is BGR
BLUE = (255, 0, 0)
RED  = (0, 0, 255)

# Labels for Script-1 (sickle state over time)
LABEL_CHANGED   = 0  # cell changed / sickled
LABEL_UNCHANGED = 1  # cell unchanged / non-sickled

# Labels for pocked classification (Script-1)
LABEL_NONPOCKED = 0
LABEL_POCKED    = 1

# Labels for Script-2 (same numeric values, clearer names)
LABEL_SICKLE    = 0  # Sickle (changed from frame 0)
LABEL_NONSICKLE = 1  # Non-sickle (unchanged from frame 0)

# Class name maps
DNAME = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
PNAME = {0: 'Non-pocked', 1: 'Pocked'}
SNAME = {0: 'Sickle',     1: 'Non-sickle'}
CLS_ID = {v: k for k, v in DNAME.items()}

# Dense trend sampling step (Script-2)
DENSE_TREND_STEP = 2

# --- Nature-style colour palettes (Script-2 plots) ---
COLOR_NS_AR   = "#2878B5"   # Science Blue  (Non-sickle / AR)
COLOR_S_AR    = "#C82423"   # Nature Red    (Sickle / AR)
PALETTE_AR    = {0: COLOR_S_AR,  1: COLOR_NS_AR,
                 'Non-sickle': COLOR_NS_AR, 'Sickle': COLOR_S_AR}

COLOR_NS_ECC  = "#9C27B0"   # Purple        (Non-sickle / ECC)
COLOR_S_ECC   = "#FF6F00"   # Deep Orange   (Sickle / ECC)
PALETTE_ECC   = {0: COLOR_S_ECC,  1: COLOR_NS_ECC,
                 'Non-sickle': COLOR_NS_ECC, 'Sickle': COLOR_S_ECC}

COLOR_NS_CIRC = "#4CAF50"   # Green         (Non-sickle / Circularity)
COLOR_S_CIRC  = "#E91E63"   # Pink          (Sickle / Circularity)
PALETTE_CIRC  = {0: COLOR_S_CIRC, 1: COLOR_NS_CIRC,
                 'Non-sickle': COLOR_NS_CIRC, 'Sickle': COLOR_S_CIRC}

# Multi-frame colour cycle
FRAME_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Seaborn / matplotlib global style (Script-2)
sns.set_theme(style="ticks", font_scale=1.2)
plt.rcParams['font.family']      = 'sans-serif'
plt.rcParams['axes.linewidth']   = 1.5

# =============================================================================
# ─── DEVICE SELECTION ─────────────────────────────────────────────────────────
# =============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)


# =============================================================================
# ─── MODEL DEFINITIONS ────────────────────────────────────────────────────────
# =============================================================================
class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.pooler_output
        return self.classifier(cls_token)


class SiameseViTChange(nn.Module):
    """Siamese ViT for change detection (Haolin's model)."""
    def __init__(self, backbone="google/vit-base-patch16-224-in21k", proj_dim=512, dropout=0.1):
        super().__init__()
        self.vit = ViTModel.from_pretrained(backbone)
        h = self.vit.config.hidden_size  # 768 for ViT-Base
        self.proj = nn.Sequential(
            nn.Linear(h, proj_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.head = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(proj_dim, 1)  # BCEWithLogitsLoss
        )

    def encode(self, x):
        x   = x.contiguous()
        out = self.vit(pixel_values=x)
        cls = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        cls = cls.contiguous()
        return self.proj(cls).contiguous()

    def forward(self, x0, x1):
        f0, f1 = self.encode(x0), self.encode(x1)
        f0 = F.normalize(f0, dim=1)
        f1 = F.normalize(f1, dim=1)
        z  = torch.cat([(f0 - f1).abs(), f0 * f1], dim=1)
        return self.head(z).reshape(-1)


# =============================================================================
# ─── MODEL LOADING ────────────────────────────────────────────────────────────
# =============================================================================
print("Loading models...")

# 7-class model (Herui's)
seven_class_model_path = 'best_model_vit_torch_macos_seven.pth'
seven_class_model = ViTClassifier(num_classes=7)
seven_class_model.load_state_dict(torch.load(seven_class_model_path, map_location=device))
seven_class_model.to(device)
seven_class_model.eval()

# General binary model (Herui's)
binary_model_path = 'best_model_vit_torch_macos_raw_vit_large_binary.pth'
binary_model = ViTClassifier(num_classes=2)
binary_model.load_state_dict(torch.load(binary_model_path, map_location=device))
binary_model.to(device)
binary_model.eval()

# Class-D binary model (Brandon's)
binary_model_path_D = "direct_vit_D.pt"
binary_model_D = ViTClassifier(num_classes=2)
binary_model_D.load_state_dict(torch.load(binary_model_path_D, map_location=device))
binary_model_D.to(device)
binary_model_D.eval()

# Class-E binary model (Brandon's)
binary_model_path_E = "direct_vit_E.pt"
binary_model_E = ViTClassifier(num_classes=2)
binary_model_E.load_state_dict(torch.load(binary_model_path_E, map_location=device))
binary_model_E.to(device)
binary_model_E.eval()

# Class-G binary model (Brandon's)
binary_model_path_Gb = "direct_vit_G.pt"
binary_model_Gb = ViTClassifier(num_classes=2)
binary_model_Gb.load_state_dict(torch.load(binary_model_path_Gb, map_location=device))
binary_model_Gb.to(device)
binary_model_Gb.eval()

# Siamese all-class change-detection model (Haolin's)
pair_model_path_All = "siamese_vit_All_Haolin.pt"
pair_model_All = SiameseViTChange()
pair_model_All.load_state_dict(torch.load(pair_model_path_All, map_location=device))
pair_model_All.to(device)
pair_model_All.eval()

# Feature extractor & transform (shared by both pipelines)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Pocked / Non-pocked binary model (Herui's) — used by Script-1 pipeline only
binary_model_path_pock = "best_model_vit_torch_macos_raw_vit_large_binary_pocked.pth"
binary_model_pock = ViTClassifier(num_classes=2)
binary_model_pock.load_state_dict(torch.load(binary_model_path_pock, map_location=device))
binary_model_pock.to(device)
binary_model_pock.eval()

print("All models loaded successfully!")


# =============================================================================
# ─── DEBUG UTILITIES (Script-1) ───────────────────────────────────────────────
# =============================================================================
DEBUG_MODE = False

def DEBUG_PRINT(msg, *args):
    if DEBUG_MODE:
        print(f"DEBUG: {msg}", *args)


# =============================================================================
# ─── SHARED HELPER FUNCTIONS ──────────────────────────────────────────────────
# =============================================================================

def remove_edge_cells(masks, threshold=0.3):
    """Remove cells that touch the frame border and are smaller than threshold * avg_area."""
    if masks.size == 0 or np.max(masks) == 0:
        print("Warning: No cells detected in masks")
        return masks.copy()
    unique_ids = range(1, np.max(masks) + 1)
    if len(unique_ids) == 0:
        return masks.copy()
    avg_cell_area = sum([np.sum(masks == cid) for cid in unique_ids]) / len(unique_ids)
    top    = masks[0, :]
    bottom = masks[-1, :]
    left   = masks[:, 0]
    right  = masks[:, -1]
    border_pixels = np.concatenate([top, bottom, left, right])
    edge_ids = np.unique(border_pixels[border_pixels > 0])
    remove_edge_list = [eid for eid in edge_ids if np.sum(masks == eid) < threshold * avg_cell_area]
    filtered_masks = np.zeros(masks.shape)
    cellidnumber = 1
    for cid in unique_ids:
        if cid not in remove_edge_list:
            filtered_masks += ((masks == cid) * cellidnumber)
            cellidnumber += 1
    return filtered_masks


def aspect_ratio(mask):
    """Aspect ratio (major / minor axis) of the dominant region in a binary mask."""
    if mask.sum() == 0:
        return 0.0
    props = regionprops(label(mask))
    if not props:
        return 0.0
    region = props[0]
    if region.minor_axis_length == 0:
        return float('inf')
    return region.major_axis_length / region.minor_axis_length


def eccentricity(mask):
    """Eccentricity [0, 1] of the dominant region (0 = circle, 1 = line)."""
    if mask.sum() == 0:
        return 0.0
    props = regionprops(label(mask))
    return props[0].eccentricity if props else 0.0


def circularity(mask):
    """Circularity = 4π·Area / Perimeter²  (capped at 1.0)."""
    if mask.sum() == 0:
        return 0.0
    props = regionprops(label(mask))
    if not props:
        return 0.0
    region = props[0]
    area, perimeter = region.area, region.perimeter
    if perimeter == 0 or area == 0:
        return 0.0
    return min((4 * np.pi * area) / (perimeter ** 2), 1.0)


def segment_frame_downscaled_ds(original_frame, model_path, out_path,
                                ratio=0.2, diameter=30,
                                is_frame_0=False,   # Script-1: triggers named debug saves
                                save_mask=False,     # Script-2: triggers per-frame mask saves
                                frame_idx=0):
    """
    Downscale → Cellpose → remove_edge_cells → upscale bboxes back to original coords.

    Unified signature covering both pipeline variants:
      - is_frame_0  (Script-1): saves BEFORE/AFTER masks under fixed filenames + .npy
      - save_mask   (Script-2): saves BEFORE/AFTER masks with frame-indexed filenames
    """
    orig_h, orig_w = original_frame.shape[:2]
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    resized_frame = cv2.resize(original_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cellpose_model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    masks, flows, styles = cellpose_model.eval(resized_frame, diameter=diameter, channels=[0, 0])

    # ----- before-edge-removal saves -----
    if is_frame_0:
        plt.imsave(out_path + "/masks_BEFORE_remove_edge_cells.png", masks, cmap="gray")
    if save_mask:
        plt.imsave(out_path + f"/masks_frame{frame_idx}_BEFORE_remove.png", masks, cmap="gray")

    masks = remove_edge_cells(masks)

    # ----- after-edge-removal saves -----
    if is_frame_0:
        DEBUG_PRINT("Saving masks.npy before remove_edge_cells")
        savepath = out_path + "/masks_remove_edge_cells.npy"
        np.save(savepath, masks)
        plt.imsave(out_path + "/masks_AFTER_remove_edge_cells.png", masks, cmap="gray")
        DEBUG_PRINT("masks.npy AFTER remove_edge_cells saved at", savepath)
    if save_mask:
        plt.imsave(out_path + f"/masks_frame{frame_idx}_AFTER_remove.png", masks, cmap="gray")

    # ----- extract & upscale bboxes -----
    unique_ids = np.unique(masks)[1:]   # skip background (0)
    bboxes = {}
    for cid in unique_ids:
        mask = (masks == cid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        x, y, w, h = cv2.boundingRect(contours[0])
        bboxes[cid] = (int(x / ratio), int(y / ratio), int(w / ratio), int(h / ratio))

    return bboxes, masks, unique_ids, resized_frame


# =============================================================================
# ─── OPTICAL FLOW & BBOX UTILITIES (Script-1 only) ───────────────────────────
# =============================================================================

def estimate_next_bboxes(prev_frame_gray, curr_frame_gray, prev_bboxes):
    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, curr_frame_gray,
                                        None, 0.5, 3, 21, 3, 5, 1.2, 0)
    updated_bboxes = {}
    for cid, bbox in prev_bboxes.items():
        x, y, w, h = bbox
        region_flow = flow[y:y + h, x:x + w]
        if region_flow.size == 0:
            updated_bboxes[cid] = bbox
            continue
        dx = np.mean(region_flow[..., 0])
        dy = np.mean(region_flow[..., 1])
        updated_bboxes[cid] = (int(x + dx), int(y + dy), w, h)
    return updated_bboxes


def resize_frame(frame, ratio):
    h, w = frame.shape[:2]
    new_size = (int(w * ratio), int(h * ratio))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def upscale_bbox(bbox, ratio):
    x, y, w, h = bbox
    return tuple(int(coord / ratio) for coord in (x, y, w, h))


# =============================================================================
# ─── CELL TRACKING FUNCTIONS ─────────────────────────────────────────────────
# =============================================================================

def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2);  ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2);  yb = min(y1 + h1, y2 + h2)
    inter_area  = max(0, xb - xa) * max(0, yb - ya)
    union_area  = w1 * h1 + w2 * h2 - inter_area
    return 0 if union_area == 0 else inter_area / union_area


def center_distance(box1, box2):
    x1, y1, *_ = box1
    x2, y2, *_ = box2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def check_size_outline(box1, box2):
    _, _, w1, h1 = box1
    _, _, w2, h2 = box2
    fold = 5
    if w2 == 0 or h2 == 0:
        return True
    c1 = 1 if (w1 > w2 and w1 - fold * w2 > 0) or (w2 > w1 and w2 - fold * w1 > 0) else 0
    c2 = 1 if (h1 > h2 and h1 - fold * h2 > 0) or (h2 > h1 and h2 - fold * h1 > 0) else 0
    return c1 == 1 and c2 == 1


def check_pos_outline_iter(box1, box2):
    x1, y1, *_ = box1
    x2, y2, *_ = box2
    return abs(x1 - x2) >= 200 and abs(y1 - y2) >= 200


def check_pos_outline(box1, box2):
    x1, y1, *_ = box1
    x2, y2, *_ = box2
    return abs(x1 - x2) >= 100 or abs(y1 - y2) >= 100


def match_cells_tracking(prev_cells, curr_masks, bboxes):
    """Match cells across frames using centre-point distance + IoU."""
    matches  = {}
    unmatched = list(np.unique(curr_masks))
    if 0 in unmatched:
        unmatched.remove(0)
    used_curr = set()

    for track_id, prev_info in prev_cells.items():
        prev_frame_index = prev_info['latest_frame_index']
        prev_box  = prev_info['bbox'][prev_frame_index]
        min_dist  = float('inf')
        best_iou  = 0
        best_cid  = None
        best_box  = None

        for cid in unmatched:
            if cid not in bboxes:
                continue
            curr_box = bboxes[cid]
            dist     = center_distance(prev_box, curr_box)
            iou      = compute_iou(prev_box, curr_box)
            if dist < min_dist:
                min_dist = dist;  best_iou = iou;  best_cid = cid;  best_box = curr_box
            elif dist < 50 and iou > best_iou:
                min_dist = dist;  best_iou = iou;  best_cid = cid;  best_box = curr_box

        if best_cid is not None:
            use_box = prev_box if (check_size_outline(prev_box, best_box) or
                                   check_pos_outline(prev_box, best_box)) else best_box
        else:
            use_box = prev_box

        matches[track_id] = {'bbox': use_box, 'class': prev_info['class']}

    return matches, used_curr


# =============================================================================
# ─── SCRIPT-1 PLOTTING HELPERS ───────────────────────────────────────────────
# =============================================================================

def plot_total_binary_ratio(df, out_path, frame_skip, fps, title='Total cell ratio (binary)'):
    """Plot (and save CSV of) overall sickle fraction over time."""
    total_pos   = df[[f'Class_{i}_pos'   for i in range(7)]].sum(axis=1)
    total_count = df[[f'Class_{i}_total' for i in range(7)]].sum(axis=1)
    total_ratio = 1 - total_pos / total_count.replace(0, np.nan)
    time_sec    = df['FrameIndex'] * frame_skip / fps
    y_percent   = total_ratio * 100

    plt.figure(figsize=(8, 5))
    plt.plot(time_sec, y_percent, label='Total sickled fraction', color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Sickled fraction (%)')
    plt.ylim(0, 100)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    csv_out_path = out_path.replace(".png", ".csv")
    pd.DataFrame({"Time_sec": time_sec,
                  "Sickled_fraction_percent (%)": y_percent}).to_csv(csv_out_path, index=False)
    print(f"Saved curve data to {csv_out_path}")


def plot_14_groups_ratio(df, out_path, frame_skip, fps,
                         title='Cell Ratio by Class and Pocked Status'):
    """Plot sickling curves for all 14 Class×Pocked combinations."""
    time_sec = df['FrameIndex'] * frame_skip / fps
    plt.figure(figsize=(14, 8))
    base_colors = ["#0922e3", "#099ae3", "#e39e09", "#9e09e3", "#895129", "#09e360", "#f05151"]

    for cls_id in range(7):
        for pock_status in [0, 1]:
            col_key   = f'Class_{cls_id}_Pock_{pock_status}'
            total_col = f'{col_key}_total'
            pos_col   = f'{col_key}_pos'
            if total_col not in df.columns or pos_col not in df.columns:
                continue
            ratio     = np.where(df[total_col] > 0,
                                 1 - df[pos_col] / df[total_col], np.nan)
            count     = int(df[total_col].max()) if len(df) > 0 else 0
            pock_lbl  = "pocked" if pock_status == 1 else "np"
            lbl       = f"{DNAME[cls_id]}-{pock_lbl} ({count} cells)"
            ls        = '-' if pock_status == 1 else '--'
            lw        = 2 if pock_status == 1 else 1.5
            plt.plot(time_sec, ratio * 100,
                     label=lbl, color=base_colors[cls_id], linestyle=ls, linewidth=lw)

    plt.xlabel('Time (s)');  plt.ylabel('Sickled fraction (%)')
    plt.ylim(0, 100);  plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True);  plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 14-group plot to {out_path}")


# =============================================================================
# ─── SCRIPT-2 STATS & PLOTTING HELPERS ───────────────────────────────────────
# =============================================================================

def get_star_string(p_value):
    """Return significance stars for a given p-value."""
    if p_value > 0.05:    return 'ns'
    elif p_value > 0.01:  return '*'
    elif p_value > 0.001: return '**'
    elif p_value > 0.0001:return '***'
    else:                 return '****'


def draw_stat_annotation(ax, x1, x2, y, h, p_val, color='k'):
    """Draw a significance bracket with star annotation on ax."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=color)
    ax.text((x1 + x2) * 0.5, y + h, get_star_string(p_val),
            ha='center', va='bottom', color=color, fontsize=12, weight='bold')


def calculate_statistics_summary(df, group_cols, metric_col):
    """Return a grouped summary table (count, mean, median, std, min, max)."""
    if df.empty:
        return pd.DataFrame()
    stats = df.groupby(group_cols)[metric_col].agg(
        Count='count', Mean='mean', Median='median',
        Std='std', Min='min', Max='max'
    ).reset_index()
    if 'Class_ID'     in stats.columns: stats['Class_Name']    = stats['Class_ID'].map(DNAME)
    if 'Sickle_Label' in stats.columns: stats['Sickle_Status'] = stats['Sickle_Label'].map(SNAME)
    return stats


# ── per-frame single-metric violin plots ──

def plot_overall_nature_style_ar(df, out_path, frame_idx=None, exclude_G=False):
    plt.figure(figsize=(5, 6))
    ax = plt.gca()
    df = df.copy()
    if exclude_G:
        df = df[df['Class_ID'] != CLS_ID['G']]
    df['Sickle_Status'] = df['Sickle_Label'].map(SNAME)
    counts = df['Sickle_Label'].value_counts().sort_index()
    n_s, n_ns = counts.get(0, 0), counts.get(1, 0)
    order = ['Non-sickle', 'Sickle']
    sns.violinplot(data=df, x='Sickle_Status', y='Aspect_Ratio', palette=PALETTE_AR,
                   inner=None, linewidth=0, alpha=0.4, ax=ax, order=order)
    sns.boxplot(data=df, x='Sickle_Status', y='Aspect_Ratio', width=0.15,
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9, 'zorder': 2},
                medianprops={'color': 'black', 'linewidth': 1.5},
                whiskerprops={'color': 'black', 'linewidth': 1.5},
                capprops={'color': 'black', 'linewidth': 1.5},
                showfliers=False, ax=ax, order=order)
    sns.stripplot(data=df, x='Sickle_Status', y='Aspect_Ratio', palette=PALETTE_AR,
                  size=3, alpha=0.6, jitter=True, zorder=1, ax=ax, order=order)
    data_s  = df[df['Sickle_Label'] == 0]['Aspect_Ratio']
    data_ns = df[df['Sickle_Label'] == 1]['Aspect_Ratio']
    if len(data_ns) > 1 and len(data_s) > 1:
        stat, p_val = mannwhitneyu(data_ns, data_s, alternative='two-sided')
        y_max = df['Aspect_Ratio'].max();  y_min = df['Aspect_Ratio'].min()
        if pd.isna(y_max) or np.isinf(y_max): y_max = 5.0
        if pd.isna(y_min) or np.isinf(y_min): y_min = 0.0
        h = y_max * 0.05
        draw_stat_annotation(ax, 0, 1, y_max + h, h * 0.5, p_val)
        if not (np.isnan(y_max + h * 5) or np.isinf(y_max + h * 5)):
            ax.set_ylim(top=y_max + h * 5)
        excl = ' (excl. G)' if exclude_G else ''
        print(f"Frame {frame_idx} Overall AR{excl} MWU: U={stat:.2f}, p={p_val:.4e}, "
              f"n_NS={len(data_ns)}, n_S={len(data_s)}")
    ax.axhline(y=1.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    ax.text(1.02, 1.6, 'AR = 1.6', transform=ax.get_yaxis_transform(),
            va='center', ha='left', fontsize=9, color='red', style='italic')
    ax.set_xlabel(""); ax.set_ylabel("Aspect Ratio", fontweight='bold')
    title = "Overall Morphology (Aspect Ratio)"
    if exclude_G:    title += " - Excluding Class G"
    if frame_idx is not None: title = f"Frame {frame_idx} - " + title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    total = n_ns + n_s
    pct_ns = n_ns / total * 100 if total > 0 else 0
    pct_s  = n_s  / total * 100 if total > 0 else 0
    ax.legend([plt.Line2D([0],[0],color=COLOR_NS_AR,lw=4),
               plt.Line2D([0],[0],color=COLOR_S_AR, lw=4)],
              [f'Non-sickle (n={n_ns}, {pct_ns:.1f}%)', f'Sickle (n={n_s}, {pct_s:.1f}%)'],
              loc='upper right', frameon=False, fontsize=10)
    sns.despine(offset=10, trim=True); plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()


def plot_7class_nature_style_ar(df, out_path, frame_idx=None):
    plt.figure(figsize=(16, 8))
    ax = plt.gca()
    df = df.copy()
    df['Sickle_Status'] = df['Sickle_Label'].map(SNAME)
    df['Class_Name']    = df['Class_ID'].map(DNAME)
    class_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    hue_order   = [1, 0]
    sns.violinplot(data=df, x='Class_Name', y='Aspect_Ratio', hue='Sickle_Label',
                   palette=PALETTE_AR, inner=None, linewidth=0, alpha=0.3,
                   order=class_order, hue_order=hue_order, dodge=True, ax=ax)
    sns.boxplot(data=df, x='Class_Name', y='Aspect_Ratio', hue='Sickle_Label',
                width=0.15, showfliers=False, dodge=True,
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9},
                order=class_order, hue_order=hue_order, ax=ax)
    sns.stripplot(data=df, x='Class_Name', y='Aspect_Ratio', hue='Sickle_Label',
                  palette=PALETTE_AR, size=2.5, alpha=0.6, dodge=True, jitter=True,
                  order=class_order, hue_order=hue_order, ax=ax)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ['Non-sickle', 'Sickle'], loc='upper right', frameon=False)
    y_max_g = df['Aspect_Ratio'].max();  y_min_g = df['Aspect_Ratio'].min()
    if pd.isna(y_max_g) or np.isinf(y_max_g): y_max_g = 5.0
    if pd.isna(y_min_g) or np.isinf(y_min_g): y_min_g = 0.0
    h_step = y_max_g * 0.05 if y_max_g > 0 else 0.1
    for i, cls_name in enumerate(class_order):
        cls_id = CLS_ID[cls_name]
        sub    = df[df['Class_ID'] == cls_id]
        d_s    = sub[sub['Sickle_Label'] == 0]['Aspect_Ratio']
        d_ns   = sub[sub['Sickle_Label'] == 1]['Aspect_Ratio']
        if len(d_ns) > 1 and len(d_s) > 1:
            try:
                stat, p_val = mannwhitneyu(d_ns, d_s, alternative='two-sided')
                draw_stat_annotation(ax, i - 0.2, i + 0.2,
                                     sub['Aspect_Ratio'].max() + h_step, h_step * 0.3, p_val)
            except ValueError as e:
                print(f"Frame {frame_idx} Class {cls_name} AR: MWU failed – {e}")
        tot = len(d_ns) + len(d_s)
        pct_ns = len(d_ns) / tot * 100 if tot else 0
        pct_s  = len(d_s)  / tot * 100 if tot else 0
        ax.text(i, y_min_g - (y_max_g - y_min_g) * 0.08,
                f'{len(d_ns)} ({pct_ns:.1f}%) | {len(d_s)} ({pct_s:.1f}%)',
                ha='center', va='top', fontsize=8, style='italic', color='gray')
    ax.text(0.5, -0.15, 'Sample counts: Non-sickle | Sickle',
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            style='italic', color='gray')
    if not any(np.isnan(v) or np.isinf(v) for v in [y_min_g, y_max_g]):
        ax.set_ylim(bottom=y_min_g - (y_max_g - y_min_g) * 0.12, top=y_max_g * 1.3)
    ax.axhline(y=1.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    ax.text(1.02, 1.6, 'AR = 1.6', transform=ax.get_yaxis_transform(),
            va='center', ha='left', fontsize=9, color='red', style='italic')
    ax.set_xlabel("Cell Class", fontweight='bold');  ax.set_ylabel("Aspect Ratio", fontweight='bold')
    title = "Morphology by Class (Aspect Ratio)"
    if frame_idx is not None: title = f"Frame {frame_idx} - " + title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    sns.despine(offset=10, trim=False); plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()


def plot_overall_nature_style_ecc(df, out_path, frame_idx=None, exclude_G=False):
    plt.figure(figsize=(5, 6))
    ax = plt.gca()
    df = df.copy()
    if exclude_G:
        df = df[df['Class_ID'] != CLS_ID['G']]
    df['Sickle_Status'] = df['Sickle_Label'].map(SNAME)
    counts = df['Sickle_Label'].value_counts().sort_index()
    n_s, n_ns = counts.get(0, 0), counts.get(1, 0)
    order = ['Non-sickle', 'Sickle']
    sns.violinplot(data=df, x='Sickle_Status', y='Eccentricity', palette=PALETTE_ECC,
                   inner=None, linewidth=0, alpha=0.4, ax=ax, order=order)
    sns.boxplot(data=df, x='Sickle_Status', y='Eccentricity', width=0.15,
                boxprops={'facecolor':'white','edgecolor':'black','alpha':0.9,'zorder':2},
                medianprops={'color':'black','linewidth':1.5},
                whiskerprops={'color':'black','linewidth':1.5},
                capprops={'color':'black','linewidth':1.5},
                showfliers=False, ax=ax, order=order)
    sns.stripplot(data=df, x='Sickle_Status', y='Eccentricity', palette=PALETTE_ECC,
                  size=3, alpha=0.6, jitter=True, zorder=1, ax=ax, order=order)
    data_s  = df[df['Sickle_Label'] == 0]['Eccentricity']
    data_ns = df[df['Sickle_Label'] == 1]['Eccentricity']
    if len(data_ns) > 1 and len(data_s) > 1:
        stat, p_val = mannwhitneyu(data_ns, data_s, alternative='two-sided')
        y_max = df['Eccentricity'].max();  y_min = df['Eccentricity'].min()
        if pd.isna(y_max) or np.isinf(y_max): y_max = 1.0
        if pd.isna(y_min) or np.isinf(y_min): y_min = 0.0
        h = y_max * 0.05
        draw_stat_annotation(ax, 0, 1, y_max + h, h * 0.5, p_val)
        if not (np.isnan(y_max + h * 6) or np.isinf(y_max + h * 6)):
            ax.set_ylim(top=y_max + h * 6)
        print(f"Frame {frame_idx} Overall ECC MWU: U={stat:.2f}, p={p_val:.4e}, "
              f"n_NS={len(data_ns)}, n_S={len(data_s)}")
    ax.set_xlabel("");  ax.set_ylabel("Eccentricity", fontweight='bold')
    title = "Overall Morphology (Eccentricity)"
    if exclude_G:       title += " - Excluding Class G"
    if frame_idx is not None: title = f"Frame {frame_idx} - " + title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    total = n_ns + n_s
    pct_ns = n_ns / total * 100 if total > 0 else 0
    pct_s  = n_s  / total * 100 if total > 0 else 0
    ax.legend([plt.Line2D([0],[0],color=COLOR_NS_ECC,lw=4),
               plt.Line2D([0],[0],color=COLOR_S_ECC, lw=4)],
              [f'Non-sickle (n={n_ns}, {pct_ns:.1f}%)', f'Sickle (n={n_s}, {pct_s:.1f}%)'],
              loc='upper right', frameon=False, fontsize=10)
    sns.despine(offset=10, trim=True); plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()


def plot_7class_nature_style_ecc(df, out_path, frame_idx=None):
    plt.figure(figsize=(16, 8))
    ax = plt.gca()
    df = df.copy()
    df['Sickle_Status'] = df['Sickle_Label'].map(SNAME)
    df['Class_Name']    = df['Class_ID'].map(DNAME)
    class_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    hue_order   = [1, 0]
    sns.violinplot(data=df, x='Class_Name', y='Eccentricity', hue='Sickle_Label',
                   palette=PALETTE_ECC, inner=None, linewidth=0, alpha=0.3,
                   order=class_order, hue_order=hue_order, dodge=True, ax=ax)
    sns.boxplot(data=df, x='Class_Name', y='Eccentricity', hue='Sickle_Label',
                width=0.15, showfliers=False, dodge=True,
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9},
                order=class_order, hue_order=hue_order, ax=ax)
    sns.stripplot(data=df, x='Class_Name', y='Eccentricity', hue='Sickle_Label',
                  palette=PALETTE_ECC, size=2.5, alpha=0.6, dodge=True, jitter=True,
                  order=class_order, hue_order=hue_order, ax=ax)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ['Non-sickle', 'Sickle'], loc='upper right', frameon=False)
    y_max_g = df['Eccentricity'].max();  y_min_g = df['Eccentricity'].min()
    if pd.isna(y_max_g) or np.isinf(y_max_g): y_max_g = 1.0
    if pd.isna(y_min_g) or np.isinf(y_min_g): y_min_g = 0.0
    h_step = y_max_g * 0.05 if y_max_g > 0 else 0.05
    for i, cls_name in enumerate(class_order):
        cls_id = CLS_ID[cls_name]
        sub    = df[df['Class_ID'] == cls_id]
        d_s    = sub[sub['Sickle_Label'] == 0]['Eccentricity']
        d_ns   = sub[sub['Sickle_Label'] == 1]['Eccentricity']
        if len(d_ns) > 1 and len(d_s) > 1:
            try:
                stat, p_val = mannwhitneyu(d_ns, d_s, alternative='two-sided')
                draw_stat_annotation(ax, i - 0.2, i + 0.2,
                                     sub['Eccentricity'].max() + h_step, h_step * 0.3, p_val)
            except ValueError as e:
                print(f"Frame {frame_idx} Class {cls_name} ECC: MWU failed – {e}")
        tot = len(d_ns) + len(d_s)
        pct_ns = len(d_ns) / tot * 100 if tot else 0
        pct_s  = len(d_s)  / tot * 100 if tot else 0
        ax.text(i, y_min_g - (y_max_g - y_min_g) * 0.08,
                f'{len(d_ns)} ({pct_ns:.1f}%) | {len(d_s)} ({pct_s:.1f}%)',
                ha='center', va='top', fontsize=8, style='italic', color='gray')
    ax.text(0.5, -0.15, 'Sample counts: Non-sickle | Sickle',
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            style='italic', color='gray')
    if not any(np.isnan(v) or np.isinf(v) for v in [y_min_g, y_max_g]):
        ax.set_ylim(bottom=y_min_g - (y_max_g - y_min_g) * 0.12, top=y_max_g * 1.3)
    ax.set_xlabel("Cell Class", fontweight='bold');  ax.set_ylabel("Eccentricity", fontweight='bold')
    title = "Morphology by Class (Eccentricity)"
    if frame_idx is not None: title = f"Frame {frame_idx} - " + title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    sns.despine(offset=10, trim=False); plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()


def plot_overall_nature_style_circ(df, out_path, frame_idx=None, exclude_G=False):
    plt.figure(figsize=(5, 6))
    ax = plt.gca()
    df = df.copy()
    if exclude_G:
        df = df[df['Class_ID'] != CLS_ID['G']]
    df['Sickle_Status'] = df['Sickle_Label'].map(SNAME)
    counts = df['Sickle_Label'].value_counts().sort_index()
    n_s, n_ns = counts.get(0, 0), counts.get(1, 0)
    order = ['Non-sickle', 'Sickle']
    sns.violinplot(data=df, x='Sickle_Status', y='Circularity', palette=PALETTE_CIRC,
                   inner=None, linewidth=0, alpha=0.4, ax=ax, order=order)
    sns.boxplot(data=df, x='Sickle_Status', y='Circularity', width=0.15,
                boxprops={'facecolor':'white','edgecolor':'black','alpha':0.9,'zorder':2},
                medianprops={'color':'black','linewidth':1.5},
                whiskerprops={'color':'black','linewidth':1.5},
                capprops={'color':'black','linewidth':1.5},
                showfliers=False, ax=ax, order=order)
    sns.stripplot(data=df, x='Sickle_Status', y='Circularity', palette=PALETTE_CIRC,
                  size=3, alpha=0.6, jitter=True, zorder=1, ax=ax, order=order)
    data_s  = df[df['Sickle_Label'] == 0]['Circularity']
    data_ns = df[df['Sickle_Label'] == 1]['Circularity']
    if len(data_ns) > 1 and len(data_s) > 1:
        stat, p_val = mannwhitneyu(data_ns, data_s, alternative='two-sided')
        y_max = df['Circularity'].max();  y_min = df['Circularity'].min()
        if pd.isna(y_max) or np.isinf(y_max): y_max = 1.0
        if pd.isna(y_min) or np.isinf(y_min): y_min = 0.0
        h = y_max * 0.05
        draw_stat_annotation(ax, 0, 1, y_max + h, h * 0.5, p_val)
        if not (np.isnan(y_max + h * 6) or np.isinf(y_max + h * 6)):
            ax.set_ylim(top=y_max + h * 6)
        excl = ' (excl. G)' if exclude_G else ''
        print(f"Frame {frame_idx} Overall Circularity{excl} MWU: U={stat:.2f}, p={p_val:.4e}, "
              f"n_NS={len(data_ns)}, n_S={len(data_s)}")
    ax.set_xlabel("");  ax.set_ylabel("Circularity", fontweight='bold')
    title = "Overall Morphology (Circularity)"
    if exclude_G:       title += " - Excluding Class G"
    if frame_idx is not None: title = f"Frame {frame_idx} - " + title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    total = n_ns + n_s
    pct_ns = n_ns / total * 100 if total > 0 else 0
    pct_s  = n_s  / total * 100 if total > 0 else 0
    ax.legend([plt.Line2D([0],[0],color=COLOR_NS_CIRC,lw=4),
               plt.Line2D([0],[0],color=COLOR_S_CIRC, lw=4)],
              [f'Non-sickle (n={n_ns}, {pct_ns:.1f}%)', f'Sickle (n={n_s}, {pct_s:.1f}%)'],
              loc='upper right', frameon=False, fontsize=10)
    sns.despine(offset=10, trim=True); plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()


def plot_7class_nature_style_circ(df, out_path, frame_idx=None):
    plt.figure(figsize=(16, 8))
    ax = plt.gca()
    df = df.copy()
    df['Sickle_Status'] = df['Sickle_Label'].map(SNAME)
    df['Class_Name']    = df['Class_ID'].map(DNAME)
    class_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    hue_order   = [1, 0]
    sns.violinplot(data=df, x='Class_Name', y='Circularity', hue='Sickle_Label',
                   palette=PALETTE_CIRC, inner=None, linewidth=0, alpha=0.3,
                   order=class_order, hue_order=hue_order, dodge=True, ax=ax)
    sns.boxplot(data=df, x='Class_Name', y='Circularity', hue='Sickle_Label',
                width=0.15, showfliers=False, dodge=True,
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9},
                order=class_order, hue_order=hue_order, ax=ax)
    sns.stripplot(data=df, x='Class_Name', y='Circularity', hue='Sickle_Label',
                  palette=PALETTE_CIRC, size=2.5, alpha=0.6, dodge=True, jitter=True,
                  order=class_order, hue_order=hue_order, ax=ax)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ['Non-sickle', 'Sickle'], loc='upper right', frameon=False)
    y_max_g = df['Circularity'].max();  y_min_g = df['Circularity'].min()
    if pd.isna(y_max_g) or np.isinf(y_max_g): y_max_g = 1.0
    if pd.isna(y_min_g) or np.isinf(y_min_g): y_min_g = 0.0
    h_step = y_max_g * 0.05 if y_max_g > 0 else 0.05
    for i, cls_name in enumerate(class_order):
        cls_id = CLS_ID[cls_name]
        sub    = df[df['Class_ID'] == cls_id]
        d_s    = sub[sub['Sickle_Label'] == 0]['Circularity']
        d_ns   = sub[sub['Sickle_Label'] == 1]['Circularity']
        if len(d_ns) > 1 and len(d_s) > 1:
            try:
                stat, p_val = mannwhitneyu(d_ns, d_s, alternative='two-sided')
                draw_stat_annotation(ax, i - 0.2, i + 0.2,
                                     sub['Circularity'].max() + h_step, h_step * 0.3, p_val)
            except ValueError as e:
                print(f"Frame {frame_idx} Class {cls_name} Circularity: MWU failed – {e}")
        tot = len(d_ns) + len(d_s)
        pct_ns = len(d_ns) / tot * 100 if tot else 0
        pct_s  = len(d_s)  / tot * 100 if tot else 0
        ax.text(i, y_min_g - (y_max_g - y_min_g) * 0.08,
                f'{len(d_ns)} ({pct_ns:.1f}%) | {len(d_s)} ({pct_s:.1f}%)',
                ha='center', va='top', fontsize=8, style='italic', color='gray')
    ax.text(0.5, -0.15, 'Sample counts: Non-sickle | Sickle',
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            style='italic', color='gray')
    if not any(np.isnan(v) or np.isinf(v) for v in [y_min_g, y_max_g]):
        ax.set_ylim(bottom=y_min_g - (y_max_g - y_min_g) * 0.12, top=y_max_g * 1.3)
    ax.set_xlabel("Cell Class", fontweight='bold');  ax.set_ylabel("Circularity", fontweight='bold')
    title = "Morphology by Class (Circularity)"
    if frame_idx is not None: title = f"Frame {frame_idx} - " + title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    sns.despine(offset=10, trim=False); plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight'); plt.close()


def _multiframe_violin_pair(all_frames_df, out_path, target_frames, metric, title_prefix):
    """Generic helper: draw Non-sickle vs Sickle across-frames violin+box plots."""
    df_filt = all_frames_df[all_frames_df['Frame_Index'].isin(target_frames)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, sickle_val, grp_title in [
        (axes[0], 1, f"Non-sickle Cells – {metric} Across Frames"),
        (axes[1], 0, f"Sickle Cells – {metric} Across Frames"),
    ]:
        sub = df_filt[df_filt['Sickle_Label'] == sickle_val]
        if not sub.empty:
            sns.violinplot(data=sub, x='Frame_Index', y=metric,
                           palette=FRAME_COLORS[:len(target_frames)],
                           inner=None, linewidth=0, alpha=0.4, ax=ax)
            sns.boxplot(data=sub, x='Frame_Index', y=metric, width=0.15,
                        boxprops={'facecolor':'white','edgecolor':'black','alpha':0.9},
                        medianprops={'color':'black','linewidth':1.5},
                        showfliers=False, ax=ax)
        if metric == 'Aspect_Ratio':
            ax.axhline(y=1.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
        ax.set_xlabel("Frame", fontweight='bold');  ax.set_ylabel(metric, fontweight='bold')
        ax.set_title(grp_title, fontsize=14, fontweight='bold')
    sns.despine(offset=10, trim=True);  plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight');  plt.close()


def plot_multiframe_comparison_ar(all_frames_df, out_path, target_frames):
    _multiframe_violin_pair(all_frames_df, out_path, target_frames, 'Aspect_Ratio', 'AR')


def plot_multiframe_comparison_ecc(all_frames_df, out_path, target_frames):
    _multiframe_violin_pair(all_frames_df, out_path, target_frames, 'Eccentricity', 'Eccentricity')


def plot_multiframe_comparison_circ(all_frames_df, out_path, target_frames):
    _multiframe_violin_pair(all_frames_df, out_path, target_frames, 'Circularity', 'Circularity')


def plot_multiframe_trend(all_frames_df, out_path, valid_target_frames, max_frame):
    """Plot Sickle-cell proportion + mean AR / ECC / Circularity trends over frames."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    captured_frames = sorted(all_frames_df['Frame_Index'].unique())

    frame_stats = []
    for frame_idx in captured_frames:
        fdf = all_frames_df[all_frames_df['Frame_Index'] == frame_idx]
        if fdf.empty:
            continue
        n_total    = len(fdf)
        n_sickle   = (fdf['Sickle_Label'] == 0).sum()
        n_nonsickle= (fdf['Sickle_Label'] == 1).sum()
        pct_sickle = n_sickle / n_total * 100 if n_total > 0 else 0

        mean_ar_s   = fdf[fdf['Sickle_Label'] == 0]['Aspect_Ratio'].mean()   if n_sickle   > 0 else np.nan
        mean_ecc_s  = fdf[fdf['Sickle_Label'] == 0]['Eccentricity'].mean()   if n_sickle   > 0 else np.nan
        mean_circ_s = fdf[fdf['Sickle_Label'] == 0]['Circularity'].mean()    if n_sickle   > 0 else np.nan

        if frame_idx == 0:
            mean_ar_s = mean_ecc_s = mean_circ_s = 0.0
            mean_ar_ns   = fdf[fdf['Sickle_Label'] == 1]['Aspect_Ratio'].mean()   if n_nonsickle > 0 else np.nan
            mean_ecc_ns  = fdf[fdf['Sickle_Label'] == 1]['Eccentricity'].mean()   if n_nonsickle > 0 else np.nan
            mean_circ_ns = fdf[fdf['Sickle_Label'] == 1]['Circularity'].mean()    if n_nonsickle > 0 else np.nan
        else:
            mean_ar_ns = mean_ecc_ns = mean_circ_ns = np.nan

        frame_stats.append({
            'Frame': frame_idx,
            'Total_Cells': n_total, 'Sickle_Count': n_sickle,
            'NonSickle_Count': n_nonsickle, 'Sickle_Percent': pct_sickle,
            'Mean_AR_Sickle': mean_ar_s,   'Mean_AR_NonSickle':   mean_ar_ns,
            'Mean_ECC_Sickle': mean_ecc_s, 'Mean_ECC_NonSickle':  mean_ecc_ns,
            'Mean_Circ_Sickle': mean_circ_s,'Mean_Circ_NonSickle': mean_circ_ns,
        })

    stats_df = pd.DataFrame(frame_stats)
    if stats_df.empty:
        plt.close();  return None

    ns_data = stats_df[stats_df['Frame'] == 0]

    ax1 = axes[0]
    ax1.plot(stats_df.dropna(subset=['Sickle_Percent'])['Frame'],
             stats_df.dropna(subset=['Sickle_Percent'])['Sickle_Percent'],
             'o-', color=COLOR_S_AR, linewidth=2, markersize=6, alpha=0.7)
    ax1.set_xlabel("Frame", fontweight='bold');  ax1.set_ylabel("Sickle Cell Percentage (%)", fontweight='bold')
    ax1.set_title("Sickle Cell Proportion Over Time", fontsize=14, fontweight='bold')

    ax2 = axes[1]
    ax2.plot(stats_df.dropna(subset=['Mean_AR_Sickle'])['Frame'],
             stats_df.dropna(subset=['Mean_AR_Sickle'])['Mean_AR_Sickle'],
             'o-', color=COLOR_S_AR, linewidth=2, markersize=6, label='Sickle', alpha=0.7)
    if not ns_data.empty:
        ax2.plot(ns_data['Frame'], ns_data['Mean_AR_NonSickle'], 's',
                 color=COLOR_NS_AR, markersize=10, label='Non-sickle (Frame 0)')
    ax2.axhline(y=1.6, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel("Frame", fontweight='bold');  ax2.set_ylabel("Mean Aspect Ratio", fontweight='bold')
    ax2.set_title("Mean AR Over Time", fontsize=14, fontweight='bold');  ax2.legend(frameon=False)

    ax3 = axes[2]
    ax3.plot(stats_df.dropna(subset=['Mean_ECC_Sickle'])['Frame'],
             stats_df.dropna(subset=['Mean_ECC_Sickle'])['Mean_ECC_Sickle'],
             'o-', color=COLOR_S_ECC, linewidth=2, markersize=6, label='Sickle', alpha=0.7)
    if not ns_data.empty:
        ax3.plot(ns_data['Frame'], ns_data['Mean_ECC_NonSickle'], 's',
                 color=COLOR_NS_ECC, markersize=10, label='Non-sickle (Frame 0)')
    ax3.set_xlabel("Frame", fontweight='bold');  ax3.set_ylabel("Mean Eccentricity", fontweight='bold')
    ax3.set_title("Mean Eccentricity Over Time", fontsize=14, fontweight='bold');  ax3.legend(frameon=False)

    ax4 = axes[3]
    ax4.plot(stats_df.dropna(subset=['Mean_Circ_Sickle'])['Frame'],
             stats_df.dropna(subset=['Mean_Circ_Sickle'])['Mean_Circ_Sickle'],
             'o-', color=COLOR_S_CIRC, linewidth=2, markersize=6, label='Sickle', alpha=0.7)
    if not ns_data.empty:
        ax4.plot(ns_data['Frame'], ns_data['Mean_Circ_NonSickle'], 's',
                 color=COLOR_NS_CIRC, markersize=10, label='Non-sickle (Frame 0)')
    ax4.set_xlabel("Frame", fontweight='bold');  ax4.set_ylabel("Mean Circularity", fontweight='bold')
    ax4.set_title("Mean Circularity Over Time", fontsize=14, fontweight='bold');  ax4.legend(frameon=False)

    sns.despine(offset=10, trim=True);  plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight');  plt.close()
    return stats_df


# =============================================================================
# ─── SCRIPT-1 PROCESSING UTILITIES ───────────────────────────────────────────
# =============================================================================

def remove_bin_label_false_positives(cell_info):
    """Smooth out isolated sickle/non-sickle flips in the state history."""
    for cid in cell_info:
        frame_indices = list(sorted(cell_info[cid]["state_history"].keys()))
        print(f"Removing false positives for {cid}")
        box_left = 0
        for frame_index in frame_indices:
            if cell_info[cid]["state_history"][frame_index] == LABEL_CHANGED:
                break
            box_left += 1

        box_right = len(frame_indices)
        for frame_index in reversed(frame_indices):
            if cell_info[cid]["state_history"][frame_index] == LABEL_UNCHANGED:
                break
            box_right -= 1

        if box_right <= box_left:
            DEBUG_PRINT("[remove_false_positive] box_right <= box_left -- no false positive detected")
            continue

        changed_count = sum(
            1 for i in range(box_left, box_right)
            if cell_info[cid]["state_history"][frame_indices[i]] == LABEL_CHANGED
        )
        new_label = LABEL_CHANGED if changed_count / (box_right - box_left) > 0.5 else LABEL_UNCHANGED
        for i in range(box_left, box_right):
            cell_info[cid]["state_history"][frame_indices[i]] = new_label


def save_intermediate_results(cell_info, df, out_path,
                               f1name="cell_info.pkl", f2name="df.pkl"):
    os.makedirs(out_path, exist_ok=True)
    cell_info_path = os.path.join(out_path, f1name)
    with open(cell_info_path, "wb") as f:
        pickle.dump(cell_info, f, protocol=pickle.HIGHEST_PROTOCOL)
    df_path = os.path.join(out_path, f2name)
    df.to_pickle(df_path, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved cell_info to {cell_info_path} and DataFrame to {df_path}")


# =============================================================================
# ─── COMBINED PROCESSING UTILITIES ───────────────────────────────────────────
# =============================================================================

def remove_bin_label_false_positives(cell_info):
    """Smooth out isolated sickle/non-sickle flips in the state history."""
    for cid in cell_info:
        frame_indices = list(sorted(cell_info[cid]["state_history"].keys()))
        print(f"Removing false positives for {cid}")
        box_left = 0
        for frame_index in frame_indices:
            if cell_info[cid]["state_history"][frame_index] == LABEL_CHANGED:
                break
            box_left += 1

        box_right = len(frame_indices)
        for frame_index in reversed(frame_indices):
            if cell_info[cid]["state_history"][frame_index] == LABEL_UNCHANGED:
                break
            box_right -= 1

        if box_right <= box_left:
            DEBUG_PRINT("[remove_false_positive] box_right <= box_left -- no false positive detected")
            continue

        changed_count = sum(
            1 for i in range(box_left, box_right)
            if cell_info[cid]["state_history"][frame_indices[i]] == LABEL_CHANGED
        )
        new_label = LABEL_CHANGED if changed_count / (box_right - box_left) > 0.5 else LABEL_UNCHANGED
        for i in range(box_left, box_right):
            cell_info[cid]["state_history"][frame_indices[i]] = new_label


def save_intermediate_results(cell_info, df, out_path,
                               f1name="cell_info.pkl", f2name="df.pkl"):
    os.makedirs(out_path, exist_ok=True)
    cell_info_path = os.path.join(out_path, f1name)
    with open(cell_info_path, "wb") as f:
        pickle.dump(cell_info, f, protocol=pickle.HIGHEST_PROTOCOL)
    df_path = os.path.join(out_path, f2name)
    df.to_pickle(df_path, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved cell_info to {cell_info_path} and DataFrame to {df_path}")


# =============================================================================
# ─── SINGLE COMBINED PROCESSING FUNCTION ─────────────────────────────────────
#
#  One pass through the video (every frame_skip frames) simultaneously:
#    • Runs Siamese model → sickle state history  (original Pipeline-1)
#    • Records AR / Eccentricity / Circularity     (original Pipeline-2)
#  After false-positive removal a second pass generates the annotated video
#  and all state-ratio statistics.  All plots from both pipelines are then
#  generated from the same cell_info / all_frames_df.
#
#  Arguments
#  ---------
#  target_frames : list[int]
#      Frame indices at which to generate morphology violin plots / stats CSVs.
#      Morphology data is actually recorded for EVERY processed frame (every
#      frame_skip-th), but detailed plots are only saved for target_frames.
# =============================================================================

def process_video_combined(video_path, out_path, video_id, output_video_path,
                           seven_class_model, binary_model, feature_extractor, transform,
                           cellpose_model_path='cyto3_train0327',
                           frame_skip=2, max_frame=480, fps=4,
                           target_frames=None):

    if target_frames is None:
        target_frames = [0, 480]
    if 0 not in target_frames:
        target_frames = [0] + target_frames
    target_frames       = sorted(target_frames)
    target_frames_set   = set(target_frames)
    save_image_frames   = {0, 480}          # frames where we save PNG images to disk

    # ── Initialise video capture ──
    print('- Initialization......')
    cap        = cv2.VideoCapture(video_path)
    fps        = cap.get(cv2.CAP_PROP_FPS)
    W          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc     = cv2.VideoWriter_fourcc(*'MJPG')
    output_fps = fps / frame_skip
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frame > total_frames:
        max_frame = total_frames

    valid_target_frames = [f for f in target_frames if f < total_frames]
    if len(valid_target_frames) < len(target_frames):
        print(f'Warning: Some target frames exceed video length ({total_frames}). '
              f'Using: {valid_target_frames}')
    print('- Initialization......Done~')

    # =========================================================================
    # FRAME 0 — segment + classify (7-class, pocked, binary) + morphology
    # =========================================================================
    ret, first_frame = cap.read()
    if not ret:
        raise ValueError("Cannot load frame 0 from the video.")
    cv2.imwrite(out_path + "/first_frame.png", first_frame)

    bboxes_f0, masks_seg_f0, unique_ids_f0, _ = segment_frame_downscaled_ds(
        first_frame, cellpose_model_path, out_path,
        is_frame_0=True, save_mask=(0 in save_image_frames), frame_idx=0)

    cell_info     = {}
    morph_records = []   # morphology rows — Sickle_Label added AFTER FP removal

    print('- Process cells in frame 0......')
    for cid in tqdm(unique_ids_f0, desc='Frame 0 cells'):
        if cid not in bboxes_f0:
            continue
        mask     = (masks_seg_f0 == cid).astype(np.uint8)
        ar_val   = aspect_ratio(mask)
        ecc_val  = eccentricity(mask)
        circ_val = circularity(mask)

        x, y, w, h = bboxes_f0[cid]
        cell_crop  = first_frame[y:y + h, x:x + w]
        if cell_crop.size == 0:
            continue
        cell_pil    = Image.fromarray(cell_crop)
        cell_tensor = transform(cell_pil).unsqueeze(0)

        with torch.no_grad():
            # 7-class
            cls_output = seven_class_model(cell_tensor.to(device))
            cls_probs  = torch.softmax(cls_output, dim=1)
            cls_id     = torch.argmax(cls_probs, dim=1).item()
            cls_prob   = cls_probs[0, cls_id].item()

            # A/G disambiguation via aspect ratio
            if cls_id == CLS_ID['A'] or cls_id == CLS_ID['G']:
                cls_id = CLS_ID['G'] if ar_val >= 1.6 else CLS_ID['A']

            # Pocked / Non-pocked
            pock_output = binary_model_pock(cell_tensor.to(device))
            pock_probs  = torch.softmax(pock_output, dim=1)
            pock_label  = torch.argmax(pock_probs, dim=1).item()
            pock_prob   = pock_probs[0, pock_label].item()

            # Sickle binary (class-specific models)
            if cls_id == CLS_ID['G']:
                bin_output = binary_model_Gb(cell_tensor.to(device))
                bin_probs  = 1 - torch.softmax(bin_output, dim=1)
            elif cls_id == CLS_ID['D']:
                bin_output = binary_model_D(cell_tensor.to(device))
                bin_probs  = 1 - torch.softmax(bin_output, dim=1)
            elif cls_id == CLS_ID['E']:
                bin_output = binary_model_E(cell_tensor.to(device))
                bin_probs  = 1 - torch.softmax(bin_output, dim=1)
            else:
                bin_output = binary_model(cell_tensor.to(device))
                bin_probs  = torch.softmax(bin_output, dim=1)
            bin_label = torch.argmax(bin_probs, dim=1).item()
            bin_prob  = bin_probs[0, bin_label].item()

        cell_info[cid] = {
            'bbox':               {0: (x, y, w, h)},
            'class':              cls_id,
            'class_prob':         cls_prob,
            'state_history':      {0: bin_label},
            'state_prob_history': {0: bin_prob},
            'latest_frame_index': 0,
            'ref_tensor':         transform(cell_pil),  # reference for Siamese
            'pair_score_ema':     None,
            'above_streak':       0,
            'pock_label':         pock_label,
            'pock_prob':          pock_prob,
            'pock_state_history': {0: bin_label},
        }

        # Morphology record for frame 0.
        # Frame 0 is the baseline: all cells are Non-sickle by definition
        # (they haven't had a chance to change yet). Sickle_Label is stored
        # directly here rather than filled in later from state_history.
        morph_records.append({
            'Cell_ID':      cid,
            'Frame_Index':  0,
            'Aspect_Ratio': ar_val,
            'Eccentricity': ecc_val,
            'Circularity':  circ_val,
            'Class_ID':     cls_id,
            'Sickle_Label': LABEL_UNCHANGED,  # frame 0 → always Non-sickle
        })

    print('- Process cells in frame 0......Done~')

    # =========================================================================
    # MAIN LOOP — frame_skip-th frames: Siamese sickle detection + morphology
    # =========================================================================
    thr = 0.7;  MIN_PERSIST = 2;  EMA_coeff = 0.5

    # Loop upper bound: must reach max_frame AND all valid target frames.
    # Original Pipeline-1 used range(1, max_frame) which excluded frame max_frame;
    # but Pipeline-2 needs to include target frames (e.g. frame 480) to have data
    # for the violin plots.  We use max_frame inclusive here.
    loop_end = max(max_frame, max(valid_target_frames) if valid_target_frames else max_frame)

    for frame_idx in tqdm(range(1, loop_end + 1), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip != 0:
            continue

        # Save raw frame image only for target frames in save_image_frames
        if frame_idx in save_image_frames:
            cv2.imwrite(out_path + f"/frame_{frame_idx}.png", frame)

        bboxes, masks, unique_ids, _ = segment_frame_downscaled_ds(
            frame, cellpose_model_path, out_path, ratio=0.1, diameter=30,
            frame_idx=frame_idx)

        matched, _ = match_cells_tracking(cell_info, masks, bboxes)

        for cid, info in matched.items():
            x, y, w, h = info['bbox']
            cls_id     = info['class']
            cell_crop  = frame[y:y + h, x:x + w]
            if cell_crop.size == 0:
                continue
            cell_pil = Image.fromarray(cell_crop)

            with torch.no_grad():
                ref_t = cell_info[cid]['ref_tensor'].to(device)
                cur_t = transform(cell_pil).to(device)
                logit = pair_model_All(ref_t.unsqueeze(0), cur_t.unsqueeze(0))
                p_chg = torch.sigmoid(logit[0]).item()

                prev  = cell_info[cid]['pair_score_ema']
                s_ema = p_chg if prev is None else (EMA_coeff * p_chg + (1 - EMA_coeff) * prev)
                cell_info[cid]['pair_score_ema'] = s_ema

                is_above = (s_ema >= thr)
                streak   = cell_info[cid].get('above_streak', 0)
                streak   = streak + 1 if is_above else 0
                cell_info[cid]['above_streak'] = streak

                if streak == MIN_PERSIST:
                    for idx in range(frame_idx - (MIN_PERSIST - 1), frame_idx):
                        cell_info[cid]['state_history'][idx]      = LABEL_CHANGED
                        cell_info[cid]['pock_state_history'][idx] = LABEL_CHANGED

                bin_label = LABEL_CHANGED if streak >= MIN_PERSIST else LABEL_UNCHANGED
                bin_prob  = float(s_ema)

            cell_info[cid]['bbox'][frame_idx]               = (x, y, w, h)
            cell_info[cid]['state_history'][frame_idx]      = bin_label
            cell_info[cid]['pock_state_history'][frame_idx] = bin_label
            cell_info[cid]['state_prob_history'][frame_idx] = bin_prob
            cell_info[cid]['latest_frame_index']            = frame_idx

            # ── Morphology: compute AR/ECC/Circularity from current masks ──
            ar_val = ecc_val = circ_val = 0.0
            for mask_cid in unique_ids:
                if mask_cid in bboxes:
                    mx, my, mw, mh = bboxes[mask_cid]
                    if abs(mx - x) < 50 and abs(my - y) < 50:
                        m        = (masks == mask_cid).astype(np.uint8)
                        ar_val   = aspect_ratio(m)
                        ecc_val  = eccentricity(m)
                        circ_val = circularity(m)
                        break
            if ar_val == 0.0:   # fallback: estimate from bbox
                ar_val   = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
                ecc_val  = 0.5
                circ_val = 0.5

            morph_records.append({
                'Cell_ID':      cid,
                'Frame_Index':  frame_idx,
                'Aspect_Ratio': ar_val,
                'Eccentricity': ecc_val,
                'Circularity':  circ_val,
                'Class_ID':     cls_id,
            })

    # =========================================================================
    # FALSE-POSITIVE REMOVAL — corrects state_history in-place
    # =========================================================================
    remove_bin_label_false_positives(cell_info)

    # =========================================================================
    # BUILD MORPHOLOGY DATAFRAME — assign corrected Sickle_Label from cell_info
    # Frame 0 rows already have Sickle_Label = LABEL_UNCHANGED set directly.
    # For all other frames, look up the FP-corrected state_history.
    # =========================================================================
    all_frames_df = pd.DataFrame(morph_records)
    if not all_frames_df.empty:
        def _get_label(row):
            if row['Frame_Index'] == 0:
                return row['Sickle_Label']   # already set as LABEL_UNCHANGED
            cid  = row['Cell_ID']
            fidx = row['Frame_Index']
            if cid in cell_info and fidx in cell_info[cid]['state_history']:
                return cell_info[cid]['state_history'][fidx]
            return LABEL_UNCHANGED
        all_frames_df['Sickle_Label'] = all_frames_df.apply(_get_label, axis=1)

    # =========================================================================
    # SECOND PASS — generate annotated video + collect state-ratio stats
    # =========================================================================
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (W, H))

    frame_stats          = []
    frame_stats_pock     = []
    frame_stats_14groups = []
    num_classes          = 7

    for frame_index in range(max_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_skip != 0:
            continue

        print(f"Collecting Results for frame {frame_index}......")
        if frame_index == 0:
            out.write(frame)

        frame_record          = {'Frame': frame_index}
        frame_record_pock     = {'Frame': frame_index}
        frame_record_14groups = {'Frame': frame_index}

        class_counts      = defaultdict(lambda: {'total': 0, 'state_1': 0})
        pock_counts_frame = defaultdict(lambda: {'total': 0, 'state_1': 0})
        class_pock_counts = defaultdict(lambda: {'total': 0, 'state_1': 0})

        annotated_frame   = frame.copy()
        annotated_frame_0 = frame.copy()

        for cid, info in cell_info.items():
            if frame_index not in info['bbox']:
                continue
            x, y, w, h  = info['bbox'][frame_index]
            bin_label    = info['state_history'][frame_index]
            color        = BLUE if bin_label == LABEL_UNCHANGED else RED
            text         = f"[{cid}] | C{info['class']} ({info['class_prob']:.2f})"
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated_frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 1)

            cid_cls   = info['class']
            plb       = info['pock_label']
            group_key = (cid_cls, plb)

            class_counts[cid_cls]['total']      += 1
            pock_counts_frame[plb]['total']      += 1
            class_pock_counts[group_key]['total']+= 1

            if bin_label == LABEL_UNCHANGED:
                class_counts[cid_cls]['state_1']      += 1
                pock_counts_frame[plb]['state_1']      += 1
                class_pock_counts[group_key]['state_1']+= 1

            if frame_index == 0:
                color_pk = BLUE if plb == LABEL_NONPOCKED else RED
                cv2.rectangle(annotated_frame_0, (x, y), (x + w, y + h), color_pk, 2)
                cv2.putText(annotated_frame_0, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_pk, 1)

        if frame_index == 0:
            cv2.imwrite(out_path + f"/frame_{frame_index}_annotated_pocked.png",
                        annotated_frame_0)

        out.write(annotated_frame)

        for cid_cls in range(num_classes):
            total = class_counts[cid_cls]['total']
            pos   = class_counts[cid_cls]['state_1']
            frame_record[f'Class_{cid_cls}']       = round(1 - pos / total if total > 0 else 0, 4)
            frame_record[f'Class_{cid_cls}_total'] = total
            frame_record[f'Class_{cid_cls}_pos']   = pos
        frame_stats.append(frame_record)

        for pid in range(2):
            total = pock_counts_frame[pid]['total']
            pos   = pock_counts_frame[pid]['state_1']
            frame_record_pock[f'Pock_{pid}']       = round(1 - pos / total if total > 0 else 0, 4)
            frame_record_pock[f'Pock_{pid}_total'] = total
            frame_record_pock[f'Pock_{pid}_pos']   = pos
        frame_stats_pock.append(frame_record_pock)

        for cid_cls in range(num_classes):
            for pock_status in [0, 1]:
                gk    = (cid_cls, pock_status)
                total = class_pock_counts[gk]['total']
                pos   = class_pock_counts[gk]['state_1']
                cp    = f'Class_{cid_cls}_Pock_{pock_status}'
                frame_record_14groups[cp]            = round(1 - pos / total if total > 0 else 0, 4)
                frame_record_14groups[f'{cp}_total'] = total
                frame_record_14groups[f'{cp}_pos']   = pos
        frame_stats_14groups.append(frame_record_14groups)

    cap.release();  out.release()
    print("Finished, results saved to:", output_video_path)

    # ── Final counters ──
    class_counts_final = Counter([info['class']      for info in cell_info.values()])
    pock_counts_final  = Counter([info['pock_label'] for info in cell_info.values()])

    # =========================================================================
    # PIPELINE-1 PER-VIDEO OUTPUTS
    # =========================================================================
    colors7      = ["#0922e3","#099ae3","#e39e09","#9e09e3","#895129","#09e360","#f05151"]
    colors_pock  = ["#0922e3", "#f05151"]

    # Pie chart
    sizes         = [class_counts_final.get(i, 0) for i in range(7)]
    total_cells   = sum(sizes)
    percentages   = [s / total_cells * 100 for s in sizes]
    legend_labels = [f'{DNAME[i]}: {sizes[i]} ({percentages[i]:.1f}%)' for i in range(7)]
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _ = ax.pie(sizes, startangle=140, colors=colors7, wedgeprops=dict(width=0.5))
    ax.legend(wedges, legend_labels, title="Classes", loc="center left",
              bbox_to_anchor=(0.92, 0.5), fontsize=10)
    ax.set_title("Class Distribution in Frame 0", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path + "/frame0_class_pie.png", dpi=300);  plt.close()

    # DataFrames
    df = pd.DataFrame(frame_stats)
    df['FrameIndex'] = range(len(df))
    df.to_csv(out_path + '/state_ratio_report.csv', index=False)

    df2 = pd.DataFrame(frame_stats_pock)
    df2['FrameIndex'] = range(len(df2))
    df2.to_csv(out_path + '/state_ratio_report_pock.csv', index=False)

    df_14groups = pd.DataFrame(frame_stats_14groups)
    df_14groups['FrameIndex'] = range(len(df_14groups))
    df_14groups.to_csv(out_path + '/state_ratio_report_14groups.csv', index=False)

    # State-ratio line chart (7 classes)
    time_sec = df['FrameIndex'] * frame_skip / fps
    plt.figure(figsize=(10, 6))
    for cid_cls in range(num_classes):
        y_pct = df[f'Class_{cid_cls}'] * 100
        count = class_counts_final.get(cid_cls, 0)
        plt.plot(time_sec, y_pct,
                 label=f'{DNAME[cid_cls]} ({count} cells)', color=colors7[cid_cls])
    plt.xlabel('Time (s)');  plt.ylabel('Sickled fraction (%)');  plt.ylim(0, 100)
    plt.legend();  plt.grid(True);  plt.tight_layout()
    plt.savefig(out_path + '/state_ratio_plot.png', dpi=300);  plt.close()

    plot_total_binary_ratio(df, out_path + '/state_ratio_plot_binary.png', frame_skip, fps)

    # Pocked line chart
    time_sec2   = df2['FrameIndex'] * frame_skip / fps
    total_pock  = sum(pock_counts_final.values())
    plt.figure(figsize=(10, 6))
    for pid in range(2):
        y_pct = df2[f'Pock_{pid}'] * 100
        count = pock_counts_final.get(pid, 0)
        ratio = count / total_pock * 100 if total_pock > 0 else 0
        plt.plot(time_sec2, y_pct,
                 label=f'{PNAME[pid]} ({count} cells, {ratio:.1f}%)',
                 color=colors_pock[pid])
    plt.xlabel('Time (s)');  plt.ylabel('Sickled fraction (%)');  plt.ylim(0, 100)
    plt.legend();  plt.grid(True);  plt.tight_layout()
    plt.savefig(out_path + '/state_ratio_plot_pocked.png', dpi=300);  plt.close()

    plot_14_groups_ratio(df_14groups, out_path + '/state_ratio_plot_14groups.png',
                         frame_skip, fps)

    # =========================================================================
    # PIPELINE-2 PER-VIDEO OUTPUTS
    # =========================================================================
    if not all_frames_df.empty:
        captured_frames = sorted(all_frames_df['Frame_Index'].unique())

        # Stats CSVs for all captured frames
        all_frames_df.to_csv(out_path + '/all_frames_raw_data.csv', index=False)
        calculate_statistics_summary(all_frames_df, ['Frame_Index', 'Sickle_Label'],
                                     'Aspect_Ratio' ).to_csv(out_path + '/all_frames_stats_ar.csv',   index=False)
        calculate_statistics_summary(all_frames_df, ['Frame_Index', 'Sickle_Label'],
                                     'Eccentricity' ).to_csv(out_path + '/all_frames_stats_ecc.csv',  index=False)
        calculate_statistics_summary(all_frames_df, ['Frame_Index', 'Sickle_Label'],
                                     'Circularity'  ).to_csv(out_path + '/all_frames_stats_circ.csv', index=False)

        # Violin plots only for target_frames that are in save_image_frames
        for frame_idx in valid_target_frames:
            if frame_idx not in save_image_frames:
                continue
            frame_df = all_frames_df[all_frames_df['Frame_Index'] == frame_idx]
            if frame_df.empty:
                continue

            pfx = out_path + f'/frame{frame_idx}'
            frame_df.to_csv(pfx + '_raw_data.csv', index=False)
            calculate_statistics_summary(frame_df, ['Sickle_Label'], 'Aspect_Ratio').to_csv(pfx + '_stats_ar.csv',   index=False)
            calculate_statistics_summary(frame_df, ['Sickle_Label'], 'Eccentricity').to_csv(pfx + '_stats_ecc.csv',  index=False)
            calculate_statistics_summary(frame_df, ['Sickle_Label'], 'Circularity' ).to_csv(pfx + '_stats_circ.csv', index=False)

            plot_overall_nature_style_ar(  frame_df.copy(), pfx + '_violin_overall_ar.png',          frame_idx)
            plot_overall_nature_style_ar(  frame_df.copy(), pfx + '_violin_overall_ar_excl_G.png',   frame_idx, exclude_G=True)
            plot_7class_nature_style_ar(   frame_df.copy(), pfx + '_violin_7class_ar.png',           frame_idx)
            plot_overall_nature_style_ecc( frame_df.copy(), pfx + '_violin_overall_ecc.png',         frame_idx)
            plot_overall_nature_style_ecc( frame_df.copy(), pfx + '_violin_overall_ecc_excl_G.png',  frame_idx, exclude_G=True)
            plot_7class_nature_style_ecc(  frame_df.copy(), pfx + '_violin_7class_ecc.png',          frame_idx)
            plot_overall_nature_style_circ(frame_df.copy(), pfx + '_violin_overall_circ.png',        frame_idx)
            plot_overall_nature_style_circ(frame_df.copy(), pfx + '_violin_overall_circ_excl_G.png', frame_idx, exclude_G=True)
            plot_7class_nature_style_circ( frame_df.copy(), pfx + '_violin_7class_circ.png',         frame_idx)
            ns  = len(frame_df[frame_df['Sickle_Label'] == LABEL_CHANGED])
            nns = len(frame_df[frame_df['Sickle_Label'] == LABEL_UNCHANGED])
            print(f'  Frame {frame_idx}: Sickle={ns}, Non-sickle={nns}')

        # Multi-frame comparison & trend (using all captured frames)
        plot_multiframe_comparison_ar(  all_frames_df, out_path + '/multiframe_comparison_ar.png',   valid_target_frames)
        plot_multiframe_comparison_ecc( all_frames_df, out_path + '/multiframe_comparison_ecc.png',  valid_target_frames)
        plot_multiframe_comparison_circ(all_frames_df, out_path + '/multiframe_comparison_circ.png', valid_target_frames)

        trend_stats = plot_multiframe_trend(all_frames_df, out_path + '/multiframe_trend.png',
                                            valid_target_frames, max(valid_target_frames))
        if trend_stats is not None:
            trend_stats.to_csv(out_path + '/multiframe_trend_stats.csv', index=False)

    return (cell_info, df, class_counts_final,
            df2, pock_counts_final, df_14groups,
            all_frames_df)


# =============================================================================
# ─── CLI ARG PARSING ──────────────────────────────────────────────────────────
# =============================================================================
print("-------------------- Parameterization --------------------")
parser = argparse.ArgumentParser(
    description='Combined sickle-cell pipeline: state-ratio + morphology (AR/ECC/Circularity).')
parser.add_argument('-i', '--inputs', type=str, required=True,
                    help='Comma-separated input video files, e.g. v1.mov,v2.mov')
parser.add_argument('-o', '--output_dir', type=str, required=True,
                    help='Output directory')
parser.add_argument('--frame_skip', type=int, default=2,
                    help='Process every Nth frame (default: 2)')
parser.add_argument('--max_frame', type=int, default=480,
                    help='Max frames to process (default: 480)')
parser.add_argument('--target_frames', type=str, default='0,480',
                    help='Comma-separated frame indices for morphology violin plots (default: 0,480)')

args = parser.parse_args()

video_paths   = args.inputs.split(',')
all_out       = args.output_dir
frame_skip    = args.frame_skip
max_frame     = args.max_frame
target_frames = [int(f.strip()) for f in args.target_frames.split(',')]
print(f"Frame skip: {frame_skip}  |  Max frame: {max_frame}  |  Target frames for plots: {target_frames}")

output   = [os.path.splitext(os.path.basename(v))[0] + '.avi' for v in video_paths]
out_path = [os.path.join(all_out, os.path.splitext(os.path.basename(v))[0]) for v in video_paths]
fps      = 4  # assumed capture fps

os.makedirs(all_out, exist_ok=True)
for op in out_path:
    os.makedirs(op, exist_ok=True)


# =============================================================================
# ─── MAIN EXECUTION — single loop over all videos ────────────────────────────
# =============================================================================
print("\n" + "=" * 60)
print("  Running combined pipeline (single pass per video)")
print("=" * 60)

all_stats             = []
all_class_counts      = Counter()
all_stats_pock        = []
all_class_counts_pock = Counter()
all_stats_14groups    = []
all_videos_df         = pd.DataFrame()

for idx, video_path in enumerate(video_paths):
    os.makedirs(out_path[idx], exist_ok=True)
    print(f"\n{'='*60}\nProcessing video {idx+1}/{len(video_paths)}: {video_path}\n{'='*60}")

    (cell_info, df, class_count,
     df2, pock_counts, df_14groups,
     video_morph_df) = process_video_combined(
        video_path=video_path,
        out_path=out_path[idx],
        video_id=f"V{idx + 1}",
        output_video_path=out_path[idx] + '/' + output[idx],
        seven_class_model=seven_class_model,
        binary_model=binary_model,
        feature_extractor=feature_extractor,
        transform=transform,
        frame_skip=frame_skip,
        max_frame=max_frame,
        fps=fps,
        target_frames=target_frames)

    all_stats.append(df)
    all_class_counts.update(class_count)
    all_stats_pock.append(df2)
    all_class_counts_pock.update(pock_counts)
    all_stats_14groups.append(df_14groups)

    if not video_morph_df.empty:
        video_morph_df['Video_ID'] = f"V{idx + 1}"
        all_videos_df = pd.concat([all_videos_df, video_morph_df], ignore_index=True)

    # Save intermediate pickles
    save_intermediate_results(cell_info, df,         out_path[idx])
    save_intermediate_results(cell_info, df2,         out_path[idx],
                              f1name="cell_info_pock.pkl",     f2name="df_pock.pkl")
    save_intermediate_results(cell_info, df_14groups, out_path[idx],
                              f1name="cell_info_14groups.pkl", f2name="df_14groups.pkl")


# =============================================================================
# ─── COMBINED OUTPUTS ─────────────────────────────────────────────────────────
# =============================================================================
print("\n" + "=" * 60)
print("  Generating combined outputs for all videos")
print("=" * 60)

def _merge_dfs(all_dfs):
    combined  = pd.concat(all_dfs, ignore_index=True)
    cols_sum  = combined.columns.difference(['Frame', 'FrameIndex'])
    summed    = combined.groupby('FrameIndex')[cols_sum].sum().reset_index()
    frame_map = combined.groupby('FrameIndex')['Frame'].first().reset_index()
    merged    = pd.merge(summed, frame_map, on='FrameIndex')
    cols_ord  = ['FrameIndex', 'Frame'] + [c for c in merged.columns
                                            if c not in ('FrameIndex', 'Frame')]
    return merged[cols_ord]


# ── Pipeline-1 combined ──
colors7 = ["#0922e3","#099ae3","#e39e09","#9e09e3","#895129","#09e360","#f05151"]

final_df = _merge_dfs(all_stats)
for cls_id in range(7):
    final_df[f'Class_{cls_id}'] = np.where(
        final_df[f'Class_{cls_id}_total'] > 0,
        1 - final_df[f'Class_{cls_id}_pos'] / final_df[f'Class_{cls_id}_total'], 0)
final_df.to_csv(all_out + '/state_ratio_report.csv', index=False)

final_df_pock = _merge_dfs(all_stats_pock)
for cls_id in range(2):
    final_df_pock[f'Pock_{cls_id}'] = np.where(
        final_df_pock[f'Pock_{cls_id}_total'] > 0,
        1 - final_df_pock[f'Pock_{cls_id}_pos'] / final_df_pock[f'Pock_{cls_id}_total'], 0)
final_df_pock.to_csv(all_out + '/state_ratio_report_pock.csv', index=False)

final_df_14groups = _merge_dfs(all_stats_14groups)
for cls_id in range(7):
    for pock_status in [0, 1]:
        col = f'Class_{cls_id}_Pock_{pock_status}'
        final_df_14groups[col] = np.where(
            final_df_14groups[f'{col}_total'] > 0,
            1 - final_df_14groups[f'{col}_pos'] / final_df_14groups[f'{col}_total'], 0)
final_df_14groups.to_csv(all_out + '/combined_state_ratio_report_14groups.csv', index=False)

time_sec = final_df['FrameIndex'] * frame_skip / fps
plt.figure(figsize=(10, 6))
for cls_id in range(7):
    y_pct = final_df[f'Class_{cls_id}'] * 100
    count = all_class_counts.get(cls_id, 0)
    plt.plot(time_sec, y_pct, label=f"{DNAME[cls_id]} ({count} cells)", color=colors7[cls_id])
plt.xlabel('Time (s)');  plt.ylabel('Sickled fraction (%)');  plt.ylim(0, 100)
plt.legend();  plt.grid(True);  plt.tight_layout()
plt.savefig(all_out + '/combined_state_ratio_plot.png', dpi=300);  plt.close()

plot_total_binary_ratio(final_df, all_out + '/state_ratio_plot_binary.png', frame_skip, fps)

time_sec2     = final_df_pock['FrameIndex'] * frame_skip / fps
total_pock_all= sum(all_class_counts_pock.values())
plt.figure(figsize=(10, 6))
for cls_id in range(2):
    y_pct = final_df_pock[f'Pock_{cls_id}'] * 100
    count = all_class_counts_pock.get(cls_id, 0)
    ratio = count / total_pock_all * 100 if total_pock_all > 0 else 0
    plt.plot(time_sec2, y_pct,
             label=f"{PNAME[cls_id]} ({count} cells, {ratio:.1f}%)",
             color=["#0922e3","#f05151"][cls_id])
plt.xlabel('Time (s)');  plt.ylabel('Sickled fraction (%)');  plt.ylim(0, 100)
plt.legend();  plt.grid(True);  plt.tight_layout()
plt.savefig(all_out + '/combined_state_ratio_plot_pock.png', dpi=300);  plt.close()

plot_14_groups_ratio(final_df_14groups,
                     all_out + '/combined_state_ratio_plot_14groups.png',
                     frame_skip, fps,
                     title='Combined Cell Ratio by Class and Pocked Status (All Videos)')

sizes         = [all_class_counts.get(i, 0) for i in range(7)]
total_cells   = sum(sizes)
percentages   = [s / total_cells * 100 for s in sizes]
legend_labels = [f'{DNAME[i]}: {sizes[i]} ({percentages[i]:.1f}%)' for i in range(7)]
fig, ax = plt.subplots(figsize=(8, 6))
wedges, _ = ax.pie(sizes, startangle=140, colors=colors7, wedgeprops=dict(width=0.5))
ax.legend(wedges, legend_labels, title="Classes", loc="center left",
          bbox_to_anchor=(0.92, 0.5), fontsize=10)
ax.set_title("Total Class Distribution in Frame 0 Across All Videos", fontsize=14)
plt.tight_layout()
plt.savefig(all_out + "/combined_frame0_class_pie.png", dpi=300);  plt.close()


# ── Pipeline-2 combined ──
if not all_videos_df.empty:
    all_videos_df.to_csv(all_out + '/combined_all_videos_raw_data.csv', index=False)
    calculate_statistics_summary(all_videos_df, ['Frame_Index', 'Sickle_Label'],
                                 'Aspect_Ratio').to_csv(all_out + '/combined_stats_ar.csv',   index=False)
    calculate_statistics_summary(all_videos_df, ['Frame_Index', 'Sickle_Label'],
                                 'Eccentricity').to_csv(all_out + '/combined_stats_ecc.csv',  index=False)
    calculate_statistics_summary(all_videos_df, ['Frame_Index', 'Sickle_Label'],
                                 'Circularity' ).to_csv(all_out + '/combined_stats_circ.csv', index=False)

    save_image_frames_combined = {480}
    print("\nGenerating combined violin plots...")
    for frame_idx in target_frames:
        if frame_idx not in save_image_frames_combined:
            continue
        frame_df = all_videos_df[all_videos_df['Frame_Index'] == frame_idx]
        if frame_df.empty:
            continue
        pfx = all_out + f'/combined_frame{frame_idx}'
        plot_overall_nature_style_ar(  frame_df.copy(), pfx + '_violin_overall_ar.png',          frame_idx)
        plot_overall_nature_style_ar(  frame_df.copy(), pfx + '_violin_overall_ar_excl_G.png',   frame_idx, exclude_G=True)
        plot_7class_nature_style_ar(   frame_df.copy(), pfx + '_violin_7class_ar.png',           frame_idx)
        plot_overall_nature_style_ecc( frame_df.copy(), pfx + '_violin_overall_ecc.png',         frame_idx)
        plot_overall_nature_style_ecc( frame_df.copy(), pfx + '_violin_overall_ecc_excl_G.png',  frame_idx, exclude_G=True)
        plot_7class_nature_style_ecc(  frame_df.copy(), pfx + '_violin_7class_ecc.png',          frame_idx)
        plot_overall_nature_style_circ(frame_df.copy(), pfx + '_violin_overall_circ.png',        frame_idx)
        plot_overall_nature_style_circ(frame_df.copy(), pfx + '_violin_overall_circ_excl_G.png', frame_idx, exclude_G=True)
        plot_7class_nature_style_circ( frame_df.copy(), pfx + '_violin_7class_circ.png',         frame_idx)
        print(f"  - Generated combined violin plots for frame {frame_idx}")

    plot_multiframe_comparison_ar(  all_videos_df, all_out + '/combined_multiframe_comparison_ar.png',   target_frames)
    plot_multiframe_comparison_ecc( all_videos_df, all_out + '/combined_multiframe_comparison_ecc.png',  target_frames)
    plot_multiframe_comparison_circ(all_videos_df, all_out + '/combined_multiframe_comparison_circ.png', target_frames)

    trend_stats = plot_multiframe_trend(all_videos_df, all_out + '/combined_multiframe_trend.png',
                                        target_frames, max(target_frames))
    if trend_stats is not None:
        trend_stats.to_csv(all_out + '/combined_multiframe_trend_stats.csv', index=False)
else:
    print("Warning: No morphology data collected from any video.")


# =============================================================================
# ─── COMPLETION SUMMARY ───────────────────────────────────────────────────────
# =============================================================================
print("\n" + "=" * 60)
print("========== 所有处理完成 ==========")
print("=" * 60)
print(f"\n单个视频结果保存在各自的 out_path 目录下:")
print("    state_ratio_report.csv          (7类 state-ratio 数据)")
print("    state_ratio_plot.png            (7类 sickling 曲线)")
print("    state_ratio_plot_binary.png     (总体 sickling 曲线)")
print("    state_ratio_report_pock.csv     (pocked/non-pocked 数据)")
print("    state_ratio_plot_pocked.png     (pocked/non-pocked 曲线)")
print("    state_ratio_report_14groups.csv (14组别详细数据)")
print("    state_ratio_plot_14groups.png   (14组别曲线图)")
print("    frame0_class_pie.png            (第0帧类别饼图)")
print("    frameX_raw_data.csv             (AR/ECC/Circ 原始数据)")
print("    frameX_violin_*.png             (形态小提琴图)")
print("    all_frames_raw_data.csv         (所有帧形态汇总)")
print("    multiframe_comparison_*.png     (多帧对比图)")
print("    multiframe_trend.png            (趋势图)")
print(f"\n合并结果保存在 {all_out}:")
print("    combined_state_ratio_report.csv / plot.png")
print("    combined_state_ratio_report_pock.csv / plot_pock.png")
print("    combined_state_ratio_report_14groups.csv / plot_14groups.png")
print("    combined_frame0_class_pie.png")
print("    combined_all_videos_raw_data.csv")
print("    combined_stats_ar/ecc/circ.csv")
print("    combined_frameX_violin_*.png")
print("    combined_multiframe_comparison_*.png")
print("    combined_multiframe_trend.png")
print(f"\n形态分析的目标帧: {target_frames}")
