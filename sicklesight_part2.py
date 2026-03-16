print("DEBUG: IMPORT - 1")
import cv2
import torch
import torch.nn as nn
import numpy as np

print("DEBUG: IMPORT - 2")
from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel

print("DEBUG: IMPORT - 3")
from PIL import Image
import os
from cellpose import models
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import torch.nn.functional as F

print("DEBUG: IMPORT - 4")
from collections import Counter
from skimage.measure import label, regionprops
import argparse

print("DEBUG: IMPORT - 5")

# ------------- Constants ----------------
DNAME = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
CLS_ID = {v: k for k, v in DNAME.items()}

# ====== Sickle vs Non-sickle (原 Pocked vs Non-pocked) ======
# 注意: 模型输出 0=Sickle(changed), 1=Non-sickle(unchanged)
SNAME = {0: 'Sickle', 1: "Non-sickle"}

# Sickle labels (与原始脚本一致)
LABEL_SICKLE = 0  # Sickle (changed from frame 0)
LABEL_NONSICKLE = 1  # Non-sickle (unchanged from frame 0)

# 【新增】定义密集采样的步长
DENSE_TREND_STEP = 2

# --- Nature Style Colors & Settings ---
# Aspect Ratio 配色
COLOR_NS_AR = "#2878B5"  # Science Blue (Non-sickle)
COLOR_S_AR = "#C82423"  # Nature Red (Sickle)
PALETTE_AR = {
    0: COLOR_S_AR,  # 0 = Sickle -> Red
    1: COLOR_NS_AR,  # 1 = Non-sickle -> Blue
    'Non-sickle': COLOR_NS_AR,
    'Sickle': COLOR_S_AR
}

# Eccentricity 配色（紫色系）
COLOR_NS_ECC = "#9C27B0"  # Purple (Non-sickle)
COLOR_S_ECC = "#FF6F00"  # Deep Orange (Sickle)
PALETTE_ECC = {
    0: COLOR_S_ECC,  # 0 = Sickle -> Orange
    1: COLOR_NS_ECC,  # 1 = Non-sickle -> Purple
    'Non-sickle': COLOR_NS_ECC,
    'Sickle': COLOR_S_ECC
}

# Circularity 配色（绿色系）
COLOR_NS_CIRC = "#4CAF50"  # Green (Non-sickle)
COLOR_S_CIRC = "#E91E63"  # Pink (Sickle)
PALETTE_CIRC = {
    0: COLOR_S_CIRC,  # 0 = Sickle -> Pink
    1: COLOR_NS_CIRC,  # 1 = Non-sickle -> Green
    'Non-sickle': COLOR_NS_CIRC,
    'Sickle': COLOR_S_CIRC
}

# 多帧配色 (用于跨帧对比图)
FRAME_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

sns.set_theme(style="ticks", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.linewidth'] = 1.5

# ============ Device ==============
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)


# ============ Model Definitions ==============
class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        cls_token = outputs.pooler_output
        return self.classifier(cls_token)


# Siamese ViT Model (from Haolin)
class SiameseViTChange(nn.Module):
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
        x = x.contiguous()
        out = self.vit(pixel_values=x)
        cls = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        cls = cls.contiguous()
        proj = self.proj(cls).contiguous()
        return proj

    def forward(self, x0, x1):
        f0, f1 = self.encode(x0), self.encode(x1)
        f0 = F.normalize(f0, dim=1)
        f1 = F.normalize(f1, dim=1)
        z = torch.cat([(f0 - f1).abs(), f0 * f1], dim=1)
        logit = self.head(z)
        return logit.reshape(-1)


# ============ Load Models ==============
print("Loading models...")

# 7-class model (Herui's)
seven_class_model_path = 'best_model_vit_torch_macos_seven.pth'
seven_class_model = ViTClassifier(num_classes=7)
seven_class_model.load_state_dict(torch.load(seven_class_model_path, map_location=device))
seven_class_model.to(device)
seven_class_model.eval()

# Binary model for general sickle classification (Herui's)
binary_model_path = 'best_model_vit_torch_macos_raw_vit_large_binary.pth'
binary_model = ViTClassifier(num_classes=2)
binary_model.load_state_dict(torch.load(binary_model_path, map_location=device))
binary_model.to(device)
binary_model.eval()

# Binary model for class D (Brandon's)
binary_model_path_D = "direct_vit_D.pt"
binary_model_D = ViTClassifier(num_classes=2)
binary_model_D.load_state_dict(torch.load(binary_model_path_D, map_location=device))
binary_model_D.to(device)
binary_model_D.eval()

# Binary model for class E (Brandon's)
binary_model_path_E = "direct_vit_E.pt"
binary_model_E = ViTClassifier(num_classes=2)
binary_model_E.load_state_dict(torch.load(binary_model_path_E, map_location=device))
binary_model_E.to(device)
binary_model_E.eval()

# Binary model for class G (Brandon's)
binary_model_path_Gb = "direct_vit_G.pt"
binary_model_Gb = ViTClassifier(num_classes=2)
binary_model_Gb.load_state_dict(torch.load(binary_model_path_Gb, map_location=device))
binary_model_Gb.to(device)
binary_model_Gb.eval()

# Siamese model for change detection (Haolin's) - used for frames > 0
pair_model_path_All = "siamese_vit_All_Haolin.pt"
pair_model_All = SiameseViTChange()
pair_model_All.load_state_dict(torch.load(pair_model_path_All, map_location=device))
pair_model_All.to(device)
pair_model_All.eval()

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

print("All models loaded successfully!")


# ============ Helper Functions ==============

def remove_edge_cells(masks, threshold=0.3):
    if masks.size == 0 or np.max(masks) == 0:
        return masks.copy()
    unique_ids = range(1, np.max(masks) + 1)
    if len(unique_ids) == 0:
        return masks.copy()
    avg_cell_area = sum([np.sum(masks == cid) for cid in unique_ids]) / len(unique_ids)
    top = masks[0, :]
    bottom = masks[-1, :]
    left = masks[:, 0]
    right = masks[:, -1]
    border_pixels = np.concatenate([top, bottom, left, right])
    edge_ids = np.unique(border_pixels[border_pixels > 0])
    remove_edge_list = []
    for edge_id in edge_ids:
        if np.sum(masks == edge_id) < threshold * avg_cell_area:
            remove_edge_list.append(edge_id)
    filtered_masks = np.zeros(masks.shape)
    cellidnumber = 1
    for cid in unique_ids:
        if cid not in remove_edge_list:
            filtered_masks += ((masks == cid) * cellidnumber)
            cellidnumber += 1
    return filtered_masks


def aspect_ratio(mask):
    if mask.sum() == 0:
        return 0.0
    labeled = label(mask)
    props = regionprops(labeled)
    if len(props) == 0:
        return 0.0
    region = props[0]
    major = region.major_axis_length
    minor = region.minor_axis_length
    if minor == 0:
        return float('inf')
    return major / minor


def eccentricity(mask):
    """计算细胞的偏心率 (Eccentricity)
    偏心率范围 [0, 1]: 0=圆形, 1=线性"""
    if mask.sum() == 0:
        return 0.0
    labeled = label(mask)
    props = regionprops(labeled)
    if len(props) == 0:
        return 0.0
    region = props[0]
    return region.eccentricity


def circularity_old(mask):
    """计算细胞的圆形度 (Circularity / Shape Factor)
    Circularity = 4π * Area / Perimeter²
    值范围 [0, 1]: 1=完美圆形, <1=不规则"""
    if mask.sum() == 0:
        return 0.0
    labeled = label(mask)
    props = regionprops(labeled)
    if len(props) == 0:
        return 0.0
    region = props[0]
    area = region.area
    perimeter = region.perimeter
    if perimeter == 0:
        return 0.0
    circ = (4 * np.pi * area) / (perimeter ** 2)
    return min(circ, 1.0)  # 限制最大值为1


def circularity(mask):
    """计算细胞的圆形度 (Circularity / Shape Factor)
    Circularity = 4π * Area / Perimeter²"""
    if mask.sum() == 0:
        return 0.0
    labeled = label(mask)
    props = regionprops(labeled)
    if len(props) == 0:
        return 0.0
    region = props[0]
    area = region.area
    perimeter = region.perimeter

    if perimeter == 0 or area == 0:
        return 0.0

    circ = (4 * np.pi * area) / (perimeter ** 2)
    return min(circ, 1.0)

def segment_frame_downscaled_ds(original_frame, model_path, out_path, ratio=0.2, diameter=30,
                                save_mask=False, frame_idx=0):
    orig_h, orig_w = original_frame.shape[:2]
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)
    resized_frame = cv2.resize(original_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    cellpose_model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    masks, flows, styles = cellpose_model.eval(resized_frame, diameter=diameter, channels=[0, 0])

    if save_mask:
        plt.imsave(out_path + f"/masks_frame{frame_idx}_BEFORE_remove.png", masks, cmap="gray")

    masks = remove_edge_cells(masks)

    if save_mask:
        plt.imsave(out_path + f"/masks_frame{frame_idx}_AFTER_remove.png", masks, cmap="gray")

    unique_ids = np.unique(masks)[1:]
    bboxes = {}
    for cid in unique_ids:
        mask = (masks == cid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        x, y, w, h = cv2.boundingRect(contours[0])
        x_orig = int(x / ratio)
        y_orig = int(y / ratio)
        w_orig = int(w / ratio)
        h_orig = int(h / ratio)
        bboxes[cid] = (x_orig, y_orig, w_orig, h_orig)

    return bboxes, masks, unique_ids, resized_frame


# ============ Cell Tracking Functions ============

def compute_iou(box1, box2):
    """计算两个边界框的IoU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area


def center_distance(box1, box2):
    """计算两个边界框中心点的距离"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    cx1, cy1 = x1, y1
    cx2, cy2 = x2, y2
    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def check_size_outline(box1, box2):
    """检查两个边界框的尺寸是否差异过大"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    fold = 5
    c1 = 0
    c2 = 0
    if w2 == 0 or h2 == 0:
        return True
    if w1 > w2:
        if w1 - fold * w2 > 0:
            c1 = 1
    else:
        if w2 - fold * w1 > 0:
            c1 = 1
    if h1 > h2:
        if h1 - fold * h2 > 0:
            c2 = 1
    else:
        if h2 - fold * h1 > 0:
            c2 = 1
    if c1 == 1 and c2 == 1:
        return True
    else:
        return False


def check_pos_outline(box1, box2):
    """检查两个边界框的位置是否差异过大"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if abs(x1 - x2) >= 100 or abs(y1 - y2) >= 100:
        return True
    else:
        return False


def match_cells_tracking(prev_cells, curr_masks, bboxes):
    """
    追踪细胞：将当前帧的细胞与之前帧的细胞进行匹配
    基于中心点距离和IoU
    """
    matches = {}
    unmatched = list(np.unique(curr_masks))
    if 0 in unmatched:
        unmatched.remove(0)  # remove background
    used_curr = set()

    for track_id, prev_info in prev_cells.items():
        prev_frame_index = prev_info['latest_frame_index']
        prev_box = prev_info['bbox'][prev_frame_index]
        min_dist = float('inf')
        best_iou = 0
        best_cid = None
        best_box = None

        for cid in unmatched:
            if cid not in bboxes:
                continue
            x, y, w, h = bboxes[cid][0], bboxes[cid][1], bboxes[cid][2], bboxes[cid][3]
            curr_box = (x, y, w, h)
            dist = center_distance(prev_box, curr_box)
            iou = compute_iou(prev_box, curr_box)

            if dist < min_dist:
                min_dist = dist
                best_iou = iou
                best_cid = cid
                best_box = curr_box
            elif dist < 50 and iou > best_iou:
                min_dist = dist
                best_iou = iou
                best_cid = cid
                best_box = curr_box

        if best_cid is not None:
            if check_size_outline(prev_box, best_box) or check_pos_outline(prev_box, best_box):
                matches[track_id] = {
                    'bbox': prev_box,
                    'class': prev_info['class'],
                }
            else:
                matches[track_id] = {
                    'bbox': best_box,
                    'class': prev_info['class'],
                }
        else:
            matches[track_id] = {
                'bbox': prev_box,
                'class': prev_info['class'],
            }

    return matches, used_curr


# ============ STATS & PLOTTING (High-Level Style) ============

def get_star_string(p_value):
    """根据P值返回显著性星号"""
    if p_value > 0.05:
        return 'ns'
    elif p_value > 0.01:
        return '*'
    elif p_value > 0.001:
        return '**'
    elif p_value > 0.0001:
        return '***'
    else:
        return '****'


def calculate_statistics_summary(df, group_cols, metric_col):
    """计算表格型统计数据（通用版本）"""
    if df.empty: return pd.DataFrame()

    stats = df.groupby(group_cols)[metric_col].agg(
        Count='count',
        Mean='mean',
        Median='median',
        Std='std',
        Min='min',
        Max='max'
    ).reset_index()

    if 'Class_ID' in stats.columns:
        stats['Class_Name'] = stats['Class_ID'].map(DNAME)
    if 'Sickle_Label' in stats.columns:
        stats['Sickle_Status'] = stats['Sickle_Label'].map(SNAME)

    return stats


def draw_stat_annotation(ax, x1, x2, y, h, p_val, color='k'):
    """在图上绘制显著性横线和星号"""
    star_str = get_star_string(p_val)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=color)
    ax.text((x1 + x2) * .5, y + h, star_str, ha='center', va='bottom', color=color, fontsize=12, weight='bold')


# ========== 单帧 Aspect Ratio 绘图函数 ==========

def plot_overall_nature_style_ar(df, out_path, frame_idx=None, exclude_G=False):
    """绘制Overall Aspect Ratio图
    exclude_G: 如果为True，则排除G类细胞"""
    plt.figure(figsize=(5, 6))
    ax = plt.gca()

    df = df.copy()
    if exclude_G:
        df = df[df['Class_ID'] != CLS_ID['G']]
    df['Sickle_Status'] = df['Sickle_Label'].map(SNAME)
    counts = df['Sickle_Label'].value_counts().sort_index()
    n_s = counts.get(0, 0)  # 0 = Sickle
    n_ns = counts.get(1, 0)  # 1 = Non-sickle

    sns.violinplot(data=df, x='Sickle_Status', y='Aspect_Ratio', palette=PALETTE_AR,
                   inner=None, linewidth=0, alpha=0.4, ax=ax, order=['Non-sickle', 'Sickle'])

    sns.boxplot(data=df, x='Sickle_Status', y='Aspect_Ratio', width=0.15,
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9, 'zorder': 2},
                medianprops={'color': 'black', 'linewidth': 1.5},
                whiskerprops={'color': 'black', 'linewidth': 1.5},
                capprops={'color': 'black', 'linewidth': 1.5},
                showfliers=False, ax=ax, order=['Non-sickle', 'Sickle'])

    sns.stripplot(data=df, x='Sickle_Status', y='Aspect_Ratio', palette=PALETTE_AR,
                  size=3, alpha=0.6, jitter=True, zorder=1, ax=ax, order=['Non-sickle', 'Sickle'])

    data_s = df[df['Sickle_Label'] == 0]['Aspect_Ratio']  # 0 = Sickle
    data_ns = df[df['Sickle_Label'] == 1]['Aspect_Ratio']  # 1 = Non-sickle



    # if len(data_ns) > 1 and len(data_s) > 1:
    #     stat, p_val = mannwhitneyu(data_ns, data_s, alternative='two-sided')
    #     y_max = df['Aspect_Ratio'].max()
    #     h = y_max * 0.05
    #     draw_stat_annotation(ax, 0, 1, y_max + h, h * 0.5, p_val)
    #     ax.set_ylim(top=y_max + h * 5)
    #     #print(f"Frame {frame_idx} Overall AR{" (excl. G)" if exclude_G else ""} MWU Test: U={stat:.2f}, p={p_val:.4e}, n_NS={len(data_ns)}, n_S={len(data_s)}")
    #     print(
    #         f"Frame {frame_idx} Overall AR{' (excl. G)' if exclude_G else ''} MWU Test: U={stat:.2f}, p={p_val:.4e}, n_NS={len(data_ns)}, n_S={len(data_s)}")
    if len(data_ns) > 1 and len(data_s) > 1:
        stat, p_val = mannwhitneyu(data_ns, data_s, alternative='two-sided')
        y_max = df['Aspect_Ratio'].max()
        y_min = df['Aspect_Ratio'].min()

        # 检查NaN/Inf
        if pd.isna(y_max) or pd.isna(y_min) or np.isinf(y_max) or np.isinf(y_min):
            y_max = 5.0
            y_min = 0.0

        h = y_max * 0.05
        draw_stat_annotation(ax, 0, 1, y_max + h, h * 0.5, p_val)

        # 再次检查计算结果
        if not (np.isnan(y_max + h * 5) or np.isinf(y_max + h * 5)):
            ax.set_ylim(top=y_max + h * 5)

        print(
            f"Frame {frame_idx} Overall AR{' (excl. G)' if exclude_G else ''} MWU Test: U={stat:.2f}, p={p_val:.4e}, n_NS={len(data_ns)}, n_S={len(data_s)}")

    ax.axhline(y=1.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    ax.text(1.02, 1.6, 'AR = 1.6', transform=ax.get_yaxis_transform(),
            va='center', ha='left', fontsize=9, color='red', style='italic')

    ax.set_xlabel("")
    ax.set_ylabel("Aspect Ratio", fontweight='bold')

    title_str = "Overall Morphology (Aspect Ratio)"
    if exclude_G:
        title_str += " - Excluding Class G"
    if frame_idx is not None:
        title_str = f"Frame {frame_idx} - " + title_str
    ax.set_title(title_str, fontsize=14, fontweight='bold', pad=15)

    total = n_ns + n_s
    pct_ns = (n_ns / total * 100) if total > 0 else 0
    pct_s = (n_s / total * 100) if total > 0 else 0
    legend_labels = [f'Non-sickle (n={n_ns}, {pct_ns:.1f}%)',
                     f'Sickle (n={n_s}, {pct_s:.1f}%)']
    legend_handles = [plt.Line2D([0], [0], color=COLOR_NS_AR, lw=4),
                      plt.Line2D([0], [0], color=COLOR_S_AR, lw=4)]
    ax.legend(legend_handles, legend_labels, loc='upper right', frameon=False, fontsize=10)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_7class_nature_style_ar(df, out_path, frame_idx=None):
    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    df = df.copy()
    df['Sickle_Status'] = df['Sickle_Label'].map(SNAME)
    df['Class_Name'] = df['Class_ID'].map(DNAME)
    class_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    # hue_order: 1=Non-sickle在左, 0=Sickle在右
    hue_order = [1, 0]

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

    handles, labels = ax.get_legend_handles_labels()
    # handles[:2]对应hue_order的顺序: [1, 0] -> ['Non-sickle', 'Sickle']
    ax.legend(handles[:2], ['Non-sickle', 'Sickle'], title="", loc='upper right', frameon=False)

    y_max_global = df['Aspect_Ratio'].max()
    y_min_global = df['Aspect_Ratio'].min()

    # 防止NaN或Inf - 需求3的修复
    if pd.isna(y_max_global) or pd.isna(y_min_global) or np.isinf(y_max_global) or np.isinf(y_min_global):
        y_max_global = 5.0
        y_min_global = 0.0

    h_step = y_max_global * 0.05 if y_max_global > 0 else 0.1

    for i, cls_name in enumerate(class_order):
        cls_id = CLS_ID[cls_name]
        subset = df[df['Class_ID'] == cls_id]

        d_s = subset[subset['Sickle_Label'] == 0]['Aspect_Ratio']  # 0 = Sickle
        d_ns = subset[subset['Sickle_Label'] == 1]['Aspect_Ratio']  # 1 = Non-sickle

        n_s = len(d_s)
        n_ns = len(d_ns)

        if n_ns > 1 and n_s > 1:
            try:
                stat, p_val = mannwhitneyu(d_ns, d_s, alternative='two-sided')
                x1 = i - 0.2
                x2 = i + 0.2
                local_y_max = subset['Aspect_Ratio'].max()
                draw_stat_annotation(ax, x1, x2, local_y_max + h_step, h_step * 0.3, p_val)
                print(f"Frame {frame_idx} Class {cls_name} AR: U={stat:.2f}, p={p_val:.4e}, n_NS={n_ns}, n_S={n_s}")
            except ValueError as e:
                print(f"Frame {frame_idx} Class {cls_name} AR: MWU test failed - {e}")

        total_cls = n_ns + n_s
        pct_ns_cls = (n_ns / total_cls * 100) if total_cls > 0 else 0
        pct_s_cls = (n_s / total_cls * 100) if total_cls > 0 else 0
        y_text = y_min_global - (y_max_global - y_min_global) * 0.08
        ax.text(i, y_text, f'{n_ns} ({pct_ns_cls:.1f}%) | {n_s} ({pct_s_cls:.1f}%)',
                ha='center', va='top', fontsize=8, style='italic', color='gray')

    ax.text(0.5, -0.15, 'Sample counts: Non-sickle | Sickle',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, style='italic', color='gray')

    # 需求3的修复：确保y_min_global和y_max_global有效
    if not (np.isnan(y_min_global) or np.isnan(y_max_global) or np.isinf(y_min_global) or np.isinf(y_max_global)):
        ax.set_ylim(bottom=y_min_global - (y_max_global - y_min_global) * 0.12,
                    top=y_max_global * 1.3)
    ax.set_xlabel("Cell Class", fontweight='bold')
    ax.set_ylabel("Aspect Ratio", fontweight='bold')

    title_str = "Morphology by Class (Aspect Ratio)"
    if frame_idx is not None:
        title_str = f"Frame {frame_idx} - " + title_str
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=20)

    ax.axhline(y=1.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    ax.text(1.02, 1.6, 'AR = 1.6', transform=ax.get_yaxis_transform(),
            va='center', ha='left', fontsize=9, color='red', style='italic')

    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# ========== 单帧 Eccentricity 绘图函数 ==========

def plot_overall_nature_style_ecc(df, out_path, frame_idx=None, exclude_G=False):
    """绘制Overall Eccentricity图
    exclude_G: 如果为True，则排除G类细胞"""
    plt.figure(figsize=(5, 6))
    ax = plt.gca()

    df = df.copy()
    if exclude_G:
        df = df[df['Class_ID'] != CLS_ID['G']]
    df['Sickle_Status'] = df['Sickle_Label'].map(SNAME)
    counts = df['Sickle_Label'].value_counts().sort_index()
    n_s = counts.get(0, 0)  # 0 = Sickle
    n_ns = counts.get(1, 0)  # 1 = Non-sickle

    sns.violinplot(data=df, x='Sickle_Status', y='Eccentricity', palette=PALETTE_ECC,
                   inner=None, linewidth=0, alpha=0.4, ax=ax, order=['Non-sickle', 'Sickle'])

    sns.boxplot(data=df, x='Sickle_Status', y='Eccentricity', width=0.15,
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9, 'zorder': 2},
                medianprops={'color': 'black', 'linewidth': 1.5},
                whiskerprops={'color': 'black', 'linewidth': 1.5},
                capprops={'color': 'black', 'linewidth': 1.5},
                showfliers=False, ax=ax, order=['Non-sickle', 'Sickle'])

    sns.stripplot(data=df, x='Sickle_Status', y='Eccentricity', palette=PALETTE_ECC,
                  size=3, alpha=0.6, jitter=True, zorder=1, ax=ax, order=['Non-sickle', 'Sickle'])

    data_s = df[df['Sickle_Label'] == 0]['Eccentricity']  # 0 = Sickle
    data_ns = df[df['Sickle_Label'] == 1]['Eccentricity']  # 1 = Non-sickle

    # if len(data_ns) > 1 and len(data_s) > 1:
    #     stat, p_val = mannwhitneyu(data_ns, data_s, alternative='two-sided')
    #     y_max = df['Eccentricity'].max()
    #     h = y_max * 0.05
    #     draw_stat_annotation(ax, 0, 1, y_max + h, h * 0.5, p_val)
    #     ax.set_ylim(top=y_max + h * 6)
    #     print(
    #         f"Frame {frame_idx} Overall ECC MWU Test: U={stat:.2f}, p={p_val:.4e}, n_NS={len(data_ns)}, n_S={len(data_s)}")
    if len(data_ns) > 1 and len(data_s) > 1:
        stat, p_val = mannwhitneyu(data_ns, data_s, alternative='two-sided')
        y_max = df['Eccentricity'].max()
        y_min = df['Eccentricity'].min()

        # 检查NaN/Inf
        if pd.isna(y_max) or pd.isna(y_min) or np.isinf(y_max) or np.isinf(y_min):
            y_max = 1.0
            y_min = 0.0

        h = y_max * 0.05
        draw_stat_annotation(ax, 0, 1, y_max + h, h * 0.5, p_val)

        # 再次检查计算结果
        if not (np.isnan(y_max + h * 6) or np.isinf(y_max + h * 6)):
            ax.set_ylim(top=y_max + h * 6)

        print(
            f"Frame {frame_idx} Overall ECC MWU Test: U={stat:.2f}, p={p_val:.4e}, n_NS={len(data_ns)}, n_S={len(data_s)}")

    ax.set_xlabel("")
    ax.set_ylabel("Eccentricity", fontweight='bold')

    title_str = "Overall Morphology (Eccentricity)"
    if exclude_G:
        title_str += " - Excluding Class G"
    if frame_idx is not None:
        title_str = f"Frame {frame_idx} - " + title_str
    ax.set_title(title_str, fontsize=14, fontweight='bold', pad=15)

    total = n_ns + n_s
    pct_ns = (n_ns / total * 100) if total > 0 else 0
    pct_s = (n_s / total * 100) if total > 0 else 0
    legend_labels = [f'Non-sickle (n={n_ns}, {pct_ns:.1f}%)',
                     f'Sickle (n={n_s}, {pct_s:.1f}%)']
    legend_handles = [plt.Line2D([0], [0], color=COLOR_NS_ECC, lw=4),
                      plt.Line2D([0], [0], color=COLOR_S_ECC, lw=4)]
    ax.legend(legend_handles, legend_labels, loc='upper right', frameon=False, fontsize=10)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_7class_nature_style_ecc(df, out_path, frame_idx=None):
    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    df = df.copy()
    df['Sickle_Status'] = df['Sickle_Label'].map(SNAME)
    df['Class_Name'] = df['Class_ID'].map(DNAME)
    class_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    # hue_order: 1=Non-sickle在左, 0=Sickle在右
    hue_order = [1, 0]

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

    handles, labels = ax.get_legend_handles_labels()
    # handles[:2]对应hue_order的顺序: [1, 0] -> ['Non-sickle', 'Sickle']
    ax.legend(handles[:2], ['Non-sickle', 'Sickle'], title="", loc='upper right', frameon=False)

    y_max_global = df['Eccentricity'].max()
    y_min_global = df['Eccentricity'].min()

    # 防止NaN或Inf - 需求3的修复
    if pd.isna(y_max_global) or pd.isna(y_min_global) or np.isinf(y_max_global) or np.isinf(y_min_global):
        y_max_global = 1.0
        y_min_global = 0.0

    h_step = y_max_global * 0.05 if y_max_global > 0 else 0.05

    for i, cls_name in enumerate(class_order):
        cls_id = CLS_ID[cls_name]
        subset = df[df['Class_ID'] == cls_id]

        d_s = subset[subset['Sickle_Label'] == 0]['Eccentricity']  # 0 = Sickle
        d_ns = subset[subset['Sickle_Label'] == 1]['Eccentricity']  # 1 = Non-sickle

        n_s = len(d_s)
        n_ns = len(d_ns)

        if n_ns > 1 and n_s > 1:
            try:
                stat, p_val = mannwhitneyu(d_ns, d_s, alternative='two-sided')
                x1 = i - 0.2
                x2 = i + 0.2
                local_y_max = subset['Eccentricity'].max()
                draw_stat_annotation(ax, x1, x2, local_y_max + h_step, h_step * 0.3, p_val)
                print(f"Frame {frame_idx} Class {cls_name} ECC: U={stat:.2f}, p={p_val:.4e}, n_NS={n_ns}, n_S={n_s}")
            except ValueError as e:
                print(f"Frame {frame_idx} Class {cls_name} ECC: MWU test failed - {e}")

        total_cls = n_ns + n_s
        pct_ns_cls = (n_ns / total_cls * 100) if total_cls > 0 else 0
        pct_s_cls = (n_s / total_cls * 100) if total_cls > 0 else 0
        y_text = y_min_global - (y_max_global - y_min_global) * 0.08
        ax.text(i, y_text, f'{n_ns} ({pct_ns_cls:.1f}%) | {n_s} ({pct_s_cls:.1f}%)',
                ha='center', va='top', fontsize=8, style='italic', color='gray')

    ax.text(0.5, -0.15, 'Sample counts: Non-sickle | Sickle',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, style='italic', color='gray')

    # 需求3的修复：确保y_min_global和y_max_global有效
    if not (np.isnan(y_min_global) or np.isnan(y_max_global) or np.isinf(y_min_global) or np.isinf(y_max_global)):
        ax.set_ylim(bottom=y_min_global - (y_max_global - y_min_global) * 0.12,
                    top=y_max_global * 1.3)
    ax.set_xlabel("Cell Class", fontweight='bold')
    ax.set_ylabel("Eccentricity", fontweight='bold')

    title_str = "Morphology by Class (Eccentricity)"
    if frame_idx is not None:
        title_str = f"Frame {frame_idx} - " + title_str
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=20)

    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# ========== 单帧 Circularity 绘图函数 (需求4) ==========

def plot_overall_nature_style_circ(df, out_path, frame_idx=None, exclude_G=False):
    """绘制Overall Circularity图
    exclude_G: 如果为True，则排除G类细胞"""
    plt.figure(figsize=(5, 6))
    ax = plt.gca()

    df = df.copy()
    if exclude_G:
        df = df[df['Class_ID'] != CLS_ID['G']]

    df['Sickle_Status'] = df['Sickle_Label'].map(SNAME)
    counts = df['Sickle_Label'].value_counts().sort_index()
    n_s = counts.get(0, 0)  # 0 = Sickle
    n_ns = counts.get(1, 0)  # 1 = Non-sickle

    sns.violinplot(data=df, x='Sickle_Status', y='Circularity', palette=PALETTE_CIRC,
                   inner=None, linewidth=0, alpha=0.4, ax=ax, order=['Non-sickle', 'Sickle'])

    sns.boxplot(data=df, x='Sickle_Status', y='Circularity', width=0.15,
                boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9, 'zorder': 2},
                medianprops={'color': 'black', 'linewidth': 1.5},
                whiskerprops={'color': 'black', 'linewidth': 1.5},
                capprops={'color': 'black', 'linewidth': 1.5},
                showfliers=False, ax=ax, order=['Non-sickle', 'Sickle'])

    sns.stripplot(data=df, x='Sickle_Status', y='Circularity', palette=PALETTE_CIRC,
                  size=3, alpha=0.6, jitter=True, zorder=1, ax=ax, order=['Non-sickle', 'Sickle'])

    data_s = df[df['Sickle_Label'] == 0]['Circularity']  # 0 = Sickle
    data_ns = df[df['Sickle_Label'] == 1]['Circularity']  # 1 = Non-sickle

    # if len(data_ns) > 1 and len(data_s) > 1:
    #     stat, p_val = mannwhitneyu(data_ns, data_s, alternative='two-sided')
    #     y_max = df['Circularity'].max()
    #     h = y_max * 0.05
    #     draw_stat_annotation(ax, 0, 1, y_max + h, h * 0.5, p_val)
    #     ax.set_ylim(top=y_max + h * 6)
    #     excl_str = " (excl. G)" if exclude_G else ""
    #     print(
    #         f"Frame {frame_idx} Overall Circularity{excl_str} MWU Test: U={stat:.2f}, p={p_val:.4e}, n_NS={len(data_ns)}, n_S={len(data_s)}")
    if len(data_ns) > 1 and len(data_s) > 1:
        stat, p_val = mannwhitneyu(data_ns, data_s, alternative='two-sided')
        y_max = df['Circularity'].max()
        y_min = df['Circularity'].min()

        # 检查NaN/Inf
        if pd.isna(y_max) or pd.isna(y_min) or np.isinf(y_max) or np.isinf(y_min):
            y_max = 1.0
            y_min = 0.0

        h = y_max * 0.05
        draw_stat_annotation(ax, 0, 1, y_max + h, h * 0.5, p_val)

        # 再次检查计算结果
        if not (np.isnan(y_max + h * 6) or np.isinf(y_max + h * 6)):
            ax.set_ylim(top=y_max + h * 6)

        excl_str = " (excl. G)" if exclude_G else ""
        print(
            f"Frame {frame_idx} Overall Circularity{excl_str} MWU Test: U={stat:.2f}, p={p_val:.4e}, n_NS={len(data_ns)}, n_S={len(data_s)}")

    ax.set_xlabel("")
    ax.set_ylabel("Circularity", fontweight='bold')

    title_str = "Overall Morphology (Circularity)"
    if exclude_G:
        title_str += " - Excluding Class G"
    if frame_idx is not None:
        title_str = f"Frame {frame_idx} - " + title_str
    ax.set_title(title_str, fontsize=14, fontweight='bold', pad=15)

    total = n_ns + n_s
    pct_ns = (n_ns / total * 100) if total > 0 else 0
    pct_s = (n_s / total * 100) if total > 0 else 0
    legend_labels = [f'Non-sickle (n={n_ns}, {pct_ns:.1f}%)',
                     f'Sickle (n={n_s}, {pct_s:.1f}%)']
    legend_handles = [plt.Line2D([0], [0], color=COLOR_NS_CIRC, lw=4),
                      plt.Line2D([0], [0], color=COLOR_S_CIRC, lw=4)]
    ax.legend(legend_handles, legend_labels, loc='upper right', frameon=False, fontsize=10)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_7class_nature_style_circ(df, out_path, frame_idx=None):
    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    df = df.copy()
    df['Sickle_Status'] = df['Sickle_Label'].map(SNAME)
    df['Class_Name'] = df['Class_ID'].map(DNAME)
    class_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

    # hue_order: 1=Non-sickle在左, 0=Sickle在右
    hue_order = [1, 0]

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

    handles, labels = ax.get_legend_handles_labels()
    # handles[:2]对应hue_order的顺序: [1, 0] -> ['Non-sickle', 'Sickle']
    ax.legend(handles[:2], ['Non-sickle', 'Sickle'], title="", loc='upper right', frameon=False)

    y_max_global = df['Circularity'].max()
    y_min_global = df['Circularity'].min()

    # 防止NaN或Inf
    if pd.isna(y_max_global) or pd.isna(y_min_global) or np.isinf(y_max_global) or np.isinf(y_min_global):
        y_max_global = 1.0
        y_min_global = 0.0

    h_step = y_max_global * 0.05 if y_max_global > 0 else 0.05

    for i, cls_name in enumerate(class_order):
        cls_id = CLS_ID[cls_name]
        subset = df[df['Class_ID'] == cls_id]

        d_s = subset[subset['Sickle_Label'] == 0]['Circularity']  # 0 = Sickle
        d_ns = subset[subset['Sickle_Label'] == 1]['Circularity']  # 1 = Non-sickle

        n_s = len(d_s)
        n_ns = len(d_ns)

        if n_ns > 1 and n_s > 1:
            try:
                stat, p_val = mannwhitneyu(d_ns, d_s, alternative='two-sided')
                x1 = i - 0.2
                x2 = i + 0.2
                local_y_max = subset['Circularity'].max()
                draw_stat_annotation(ax, x1, x2, local_y_max + h_step, h_step * 0.3, p_val)
                print(
                    f"Frame {frame_idx} Class {cls_name} Circularity: U={stat:.2f}, p={p_val:.4e}, n_NS={n_ns}, n_S={n_s}")
            except ValueError as e:
                print(f"Frame {frame_idx} Class {cls_name} Circularity: MWU test failed - {e}")

        total_cls = n_ns + n_s
        pct_ns_cls = (n_ns / total_cls * 100) if total_cls > 0 else 0
        pct_s_cls = (n_s / total_cls * 100) if total_cls > 0 else 0
        y_text = y_min_global - (y_max_global - y_min_global) * 0.08
        ax.text(i, y_text, f'{n_ns} ({pct_ns_cls:.1f}%) | {n_s} ({pct_s_cls:.1f}%)',
                ha='center', va='top', fontsize=8, style='italic', color='gray')

    ax.text(0.5, -0.15, 'Sample counts: Non-sickle | Sickle',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, style='italic', color='gray')

    if not (np.isnan(y_min_global) or np.isnan(y_max_global) or np.isinf(y_min_global) or np.isinf(y_max_global)):
        ax.set_ylim(bottom=y_min_global - (y_max_global - y_min_global) * 0.12,
                    top=y_max_global * 1.3)

    ax.set_xlabel("Cell Class", fontweight='bold')
    ax.set_ylabel("Circularity", fontweight='bold')

    title_str = "Morphology by Class (Circularity)"
    if frame_idx is not None:
        title_str = f"Frame {frame_idx} - " + title_str
    ax.set_title(title_str, fontsize=16, fontweight='bold', pad=20)

    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiframe_comparison_circ(all_frames_df, out_path, target_frames):
    """绘制多帧对比的Circularity图 (需求4)"""
    all_frames_df = all_frames_df[all_frames_df['Frame_Index'].isin(target_frames)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：Non-sickle 跨帧对比 (Label = 1)
    ax1 = axes[0]
    df_ns = all_frames_df[all_frames_df['Sickle_Label'] == 1]  # 1 = Non-sickle

    if not df_ns.empty:
        sns.violinplot(data=df_ns, x='Frame_Index', y='Circularity',
                       palette=FRAME_COLORS[:len(target_frames)],
                       inner=None, linewidth=0, alpha=0.4, ax=ax1)
        sns.boxplot(data=df_ns, x='Frame_Index', y='Circularity', width=0.15,
                    boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9},
                    medianprops={'color': 'black', 'linewidth': 1.5},
                    showfliers=False, ax=ax1)

    ax1.set_xlabel("Frame", fontweight='bold')
    ax1.set_ylabel("Circularity", fontweight='bold')
    ax1.set_title("Non-sickle Cells - Circularity Across Frames", fontsize=14, fontweight='bold')

    # 右图：Sickle 跨帧对比 (Label = 0)
    ax2 = axes[1]
    df_s = all_frames_df[all_frames_df['Sickle_Label'] == 0]  # 0 = Sickle

    if not df_s.empty:
        sns.violinplot(data=df_s, x='Frame_Index', y='Circularity',
                       palette=FRAME_COLORS[:len(target_frames)],
                       inner=None, linewidth=0, alpha=0.4, ax=ax2)
        sns.boxplot(data=df_s, x='Frame_Index', y='Circularity', width=0.15,
                    boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9},
                    medianprops={'color': 'black', 'linewidth': 1.5},
                    showfliers=False, ax=ax2)

    ax2.set_xlabel("Frame", fontweight='bold')
    ax2.set_ylabel("Circularity", fontweight='bold')
    ax2.set_title("Sickle Cells - Circularity Across Frames", fontsize=14, fontweight='bold')

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


# ========== 多帧对比图 ==========

def plot_multiframe_comparison_ar(all_frames_df, out_path, target_frames):
    """绘制多帧对比的Aspect Ratio图"""
    all_frames_df = all_frames_df[all_frames_df['Frame_Index'].isin(target_frames)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：Non-sickle 跨帧对比 (Label = 1)
    ax1 = axes[0]
    df_ns = all_frames_df[all_frames_df['Sickle_Label'] == 1]  # 1 = Non-sickle

    if not df_ns.empty:
        sns.violinplot(data=df_ns, x='Frame_Index', y='Aspect_Ratio',
                       palette=FRAME_COLORS[:len(target_frames)],
                       inner=None, linewidth=0, alpha=0.4, ax=ax1)
        sns.boxplot(data=df_ns, x='Frame_Index', y='Aspect_Ratio', width=0.15,
                    boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9},
                    medianprops={'color': 'black', 'linewidth': 1.5},
                    showfliers=False, ax=ax1)

    ax1.axhline(y=1.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    ax1.set_xlabel("Frame", fontweight='bold')
    ax1.set_ylabel("Aspect Ratio", fontweight='bold')
    ax1.set_title("Non-sickle Cells - AR Across Frames", fontsize=14, fontweight='bold')

    # 右图：Sickle 跨帧对比 (Label = 0)
    ax2 = axes[1]
    df_s = all_frames_df[all_frames_df['Sickle_Label'] == 0]  # 0 = Sickle

    if not df_s.empty:
        sns.violinplot(data=df_s, x='Frame_Index', y='Aspect_Ratio',
                       palette=FRAME_COLORS[:len(target_frames)],
                       inner=None, linewidth=0, alpha=0.4, ax=ax2)
        sns.boxplot(data=df_s, x='Frame_Index', y='Aspect_Ratio', width=0.15,
                    boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9},
                    medianprops={'color': 'black', 'linewidth': 1.5},
                    showfliers=False, ax=ax2)

    ax2.axhline(y=1.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    ax2.set_xlabel("Frame", fontweight='bold')
    ax2.set_ylabel("Aspect Ratio", fontweight='bold')
    ax2.set_title("Sickle Cells - AR Across Frames", fontsize=14, fontweight='bold')

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiframe_comparison_ecc(all_frames_df, out_path, target_frames):
    """绘制多帧对比的Eccentricity图"""
    all_frames_df = all_frames_df[all_frames_df['Frame_Index'].isin(target_frames)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：Non-sickle 跨帧对比 (Label = 1)
    ax1 = axes[0]
    df_ns = all_frames_df[all_frames_df['Sickle_Label'] == 1]  # 1 = Non-sickle

    if not df_ns.empty:
        sns.violinplot(data=df_ns, x='Frame_Index', y='Eccentricity',
                       palette=FRAME_COLORS[:len(target_frames)],
                       inner=None, linewidth=0, alpha=0.4, ax=ax1)
        sns.boxplot(data=df_ns, x='Frame_Index', y='Eccentricity', width=0.15,
                    boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9},
                    medianprops={'color': 'black', 'linewidth': 1.5},
                    showfliers=False, ax=ax1)

    ax1.set_xlabel("Frame", fontweight='bold')
    ax1.set_ylabel("Eccentricity", fontweight='bold')
    ax1.set_title("Non-sickle Cells - Eccentricity Across Frames", fontsize=14, fontweight='bold')

    # 右图：Sickle 跨帧对比 (Label = 0)
    ax2 = axes[1]
    df_s = all_frames_df[all_frames_df['Sickle_Label'] == 0]  # 0 = Sickle

    if not df_s.empty:
        sns.violinplot(data=df_s, x='Frame_Index', y='Eccentricity',
                       palette=FRAME_COLORS[:len(target_frames)],
                       inner=None, linewidth=0, alpha=0.4, ax=ax2)
        sns.boxplot(data=df_s, x='Frame_Index', y='Eccentricity', width=0.15,
                    boxprops={'facecolor': 'white', 'edgecolor': 'black', 'alpha': 0.9},
                    medianprops={'color': 'black', 'linewidth': 1.5},
                    showfliers=False, ax=ax2)

    ax2.set_xlabel("Frame", fontweight='bold')
    ax2.set_ylabel("Eccentricity", fontweight='bold')
    ax2.set_title("Sickle Cells - Eccentricity Across Frames", fontsize=14, fontweight='bold')

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_multiframe_trend(all_frames_df, out_path, valid_target_frames, max_frame):
    """绘制Sickle细胞比例随帧变化的趋势图
    需求5: non-sickle只显示第0帧，sickle显示0帧+指定帧+密集采样点"""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # 需求5: 生成采样帧
    # Non-sickle: 只在frame 0
    # Sickle: frame 0 + valid_target_frames + 密集采样

    # 确定sickle的采样帧
    sickle_frames = set(valid_target_frames)

    # 如果max_frame很大，增加密集采样点
    if max_frame > 100:
        # 每2帧一个采样点（不包括已经在valid_target_frames中的）
        dense_frames = list(range(0, max_frame + 1, 2))
        sickle_frames.update(dense_frames)

    sickle_frames = sorted(list(sickle_frames))
    captured_frames = sorted(all_frames_df['Frame_Index'].unique())

    # 计算每帧的统计数据
    frame_stats = []
    for frame_idx in captured_frames:
        frame_df = all_frames_df[all_frames_df['Frame_Index'] == frame_idx]
        if frame_df.empty:
            continue

        n_total = len(frame_df)
        n_sickle = (frame_df['Sickle_Label'] == 0).sum()  # 0 = Sickle
        n_nonsickle = (frame_df['Sickle_Label'] == 1).sum()  # 1 = Non-sickle
        pct_sickle = (n_sickle / n_total * 100) if n_total > 0 else 0

        # 只计算sickle细胞的均值（non-sickle只在frame 0有点）
        mean_ar_s = frame_df[frame_df['Sickle_Label'] == 0]['Aspect_Ratio'].mean() if n_sickle > 0 else np.nan
        mean_ecc_s = frame_df[frame_df['Sickle_Label'] == 0]['Eccentricity'].mean() if n_sickle > 0 else np.nan
        mean_circ_s = frame_df[frame_df['Sickle_Label'] == 0]['Circularity'].mean() if n_sickle > 0 else np.nan

        # Non-sickle只在frame 0计算
        if frame_idx == 0:
            # Sickle (红线): 强制设为 0 值，从坐标轴底部出发
            mean_ar_s = 0.0
            mean_ecc_s = 0.0
            mean_circ_s = 0.0
            
            mean_ar_ns = frame_df[frame_df['Sickle_Label'] == 1]['Aspect_Ratio'].mean() if n_nonsickle > 0 else np.nan
            mean_ecc_ns = frame_df[frame_df['Sickle_Label'] == 1]['Eccentricity'].mean() if n_nonsickle > 0 else np.nan
            mean_circ_ns = frame_df[frame_df['Sickle_Label'] == 1]['Circularity'].mean() if n_nonsickle > 0 else np.nan
        else:
            mean_ar_ns = np.nan
            mean_ecc_ns = np.nan
            mean_circ_ns = np.nan

        frame_stats.append({
            'Frame': frame_idx,
            'Total_Cells': n_total,
            'Sickle_Count': n_sickle,
            'NonSickle_Count': n_nonsickle,
            'Sickle_Percent': pct_sickle,
            'Mean_AR_Sickle': mean_ar_s,
            'Mean_AR_NonSickle': mean_ar_ns,
            'Mean_ECC_Sickle': mean_ecc_s,
            'Mean_ECC_NonSickle': mean_ecc_ns,
            'Mean_Circ_Sickle': mean_circ_s,
            'Mean_Circ_NonSickle': mean_circ_ns
        })

    stats_df = pd.DataFrame(frame_stats)

    if stats_df.empty:
        plt.close()
        return None

    # 图1：Sickle比例趋势
    ax1 = axes[0]
    data_pct = stats_df.dropna(subset=['Sickle_Percent'])
    ax1.plot(data_pct['Frame'], data_pct['Sickle_Percent'], 'o-',
             color=COLOR_S_AR, linewidth=2, markersize=6, label='Sickle %', alpha=0.7)
    # ax1.plot(stats_df['Frame'], stats_df['Sickle_Percent'], 'o-',
    #          color=COLOR_S_AR, linewidth=2, markersize=6, label='Sickle %', alpha=0.7)
    ax1.set_xlabel("Frame", fontweight='bold')
    ax1.set_ylabel("Sickle Cell Percentage (%)", fontweight='bold')
    ax1.set_title("Sickle Cell Proportion Over Time", fontsize=14, fontweight='bold')
    ax1.legend(frameon=False)

    # 图2：Mean AR 趋势
    ax2 = axes[1]
    # Sickle: 所有采样点
    data_ar = stats_df.dropna(subset=['Mean_AR_Sickle'])
    ax2.plot(data_ar['Frame'], data_ar['Mean_AR_Sickle'], 'o-',
             color=COLOR_S_AR, linewidth=2, markersize=6, label='Sickle', alpha=0.7)
    # ax2.plot(stats_df['Frame'], stats_df['Mean_AR_Sickle'], 'o-',
    #          color=COLOR_S_AR, linewidth=2, markersize=6, label='Sickle', alpha=0.7)
    # Non-sickle: 只显示frame 0的点
    ns_data = stats_df[stats_df['Frame'] == 0]
    if not ns_data.empty:
        ax2.plot(ns_data['Frame'], ns_data['Mean_AR_NonSickle'], 's',
                 color=COLOR_NS_AR, markersize=10, label='Non-sickle (Frame 0)')

    ax2.axhline(y=1.6, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel("Frame", fontweight='bold')
    ax2.set_ylabel("Mean Aspect Ratio", fontweight='bold')
    ax2.set_title("Mean AR Over Time", fontsize=14, fontweight='bold')
    ax2.legend(frameon=False)

    # 图3：Mean Eccentricity 趋势
    ax3 = axes[2]
    data_ecc = stats_df.dropna(subset=['Mean_ECC_Sickle'])
    ax3.plot(data_ecc['Frame'], data_ecc['Mean_ECC_Sickle'], 'o-',
             color=COLOR_S_ECC, linewidth=2, markersize=6, label='Sickle', alpha=0.7)
    # ax3.plot(stats_df['Frame'], stats_df['Mean_ECC_Sickle'], 'o-',
    #          color=COLOR_S_ECC, linewidth=2, markersize=6, label='Sickle', alpha=0.7)
    if not ns_data.empty:
        ax3.plot(ns_data['Frame'], ns_data['Mean_ECC_NonSickle'], 's',
                 color=COLOR_NS_ECC, markersize=10, label='Non-sickle (Frame 0)')
    ax3.set_xlabel("Frame", fontweight='bold')
    ax3.set_ylabel("Mean Eccentricity", fontweight='bold')
    ax3.set_title("Mean Eccentricity Over Time", fontsize=14, fontweight='bold')
    ax3.legend(frameon=False)

    # 图4：Mean Circularity 趋势 (需求4)
    ax4 = axes[3]
    data_circ = stats_df.dropna(subset=['Mean_Circ_Sickle'])
    ax4.plot(data_circ['Frame'], data_circ['Mean_Circ_Sickle'], 'o-',
             color=COLOR_S_CIRC, linewidth=2, markersize=6, label='Sickle', alpha=0.7)
    # ax4.plot(stats_df['Frame'], stats_df['Mean_Circ_Sickle'], 'o-',
    #          color=COLOR_S_CIRC, linewidth=2, markersize=6, label='Sickle', alpha=0.7)
    if not ns_data.empty:
        ax4.plot(ns_data['Frame'], ns_data['Mean_Circ_NonSickle'], 's',
                 color=COLOR_NS_CIRC, markersize=10, label='Non-sickle (Frame 0)')
    ax4.set_xlabel("Frame", fontweight='bold')
    ax4.set_ylabel("Mean Circularity", fontweight='bold')
    ax4.set_title("Mean Circularity Over Time", fontsize=14, fontweight='bold')
    ax4.legend(frameon=False)

    sns.despine(offset=10, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    return stats_df


# -------------------- Processing Functions ---------------------

def process_video_multiframe(video_path, out_path, transform, target_frames,
                             cellpose_model_path='cyto3_train0327'):
    """
    处理视频的多个指定帧
    逻辑与原始脚本完全一致：
    - Frame 0: 所有细胞初始化为 Non-sickle (LABEL_NONSICKLE=1)
    - Frame 1 到 max(target_frames): 逐帧处理，累积 EMA 和 streak
    - 只在 target_frames 输出统计和图表
    """
    print(f'Processing video: {video_path}')
    print(f'Target frames: {target_frames}')

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total frames in video: {total_frames}')

    # 确保0在target_frames中且排序
    if 0 not in target_frames:
        target_frames = [0] + target_frames
    target_frames = sorted(target_frames)

    # 过滤超出范围的帧
    valid_target_frames = [f for f in target_frames if f < total_frames]
    if len(valid_target_frames) < len(target_frames):
        print(f'Warning: Some target frames exceed video length. Using: {valid_target_frames}')

    max_target_frame = max(valid_target_frames)
    target_frames_set = set(valid_target_frames)

    # 需求1: 定义需要保存图像的帧
    save_image_frames = {0,480}

    all_frames_df = pd.DataFrame()
    cell_info = {}  # 存储细胞信息和ref_tensor

    # Siamese 模型参数 (与原脚本一致)
    thr = 0.7
    MIN_PERSIST = 2
    EMA_coeff = 0.5

    # ========== 处理 Frame 0 (初始化，所有细胞为 Non-sickle) ==========
    print(f'\n- Processing Frame 0 (initialization, all cells = Non-sickle)...')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()

    if not ret:
        print("Error: Cannot read frame 0")
        cap.release()
        return pd.DataFrame()

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    # 需求1: 只在指定帧保存图像
    if 0 in save_image_frames:
        cv2.imwrite(out_path + "/frame_0.png", first_frame)

    # 分割Frame 0
    bboxes_f0, masks_seg_f0, unique_ids_f0, _ = segment_frame_downscaled_ds(
        first_frame_rgb, cellpose_model_path, out_path,
        save_mask=(0 in save_image_frames), frame_idx=0
    )

    frame0_records = []

    for cid in tqdm(unique_ids_f0, desc='  Frame 0 Cells'):
        mask = (masks_seg_f0 == cid).astype(np.uint8)
        ar_val = aspect_ratio(mask)
        ecc_val = eccentricity(mask)
        circ_val = circularity(mask)  # 需求4: 增加circularity

        if cid not in bboxes_f0:
            continue

        x, y, w, h = bboxes_f0[cid][0], bboxes_f0[cid][1], bboxes_f0[cid][2], bboxes_f0[cid][3]
        cell_crop = first_frame_rgb[y:y + h, x:x + w]

        if cell_crop.size == 0:
            continue

        cell_pil = Image.fromarray(cell_crop)
        cell_tensor = transform(cell_pil).unsqueeze(0)

        with torch.no_grad():
            # 7-class 分类
            cls_output = seven_class_model(cell_tensor.to(device))
            cls_probs = torch.softmax(cls_output, dim=1)
            cls_id = torch.argmax(cls_probs, dim=1).item()

            # Apply AR threshold for A/G
            if cls_id == CLS_ID['A'] or cls_id == CLS_ID['G']:
                if ar_val >= 1.6:
                    cls_id = CLS_ID['G']
                else:
                    cls_id = CLS_ID['A']

        # Frame 0: 所有细胞默认为 Non-sickle (与原脚本一致)
        sickle_label = LABEL_NONSICKLE  # = 1

        # 存储cell_info用于后续帧追踪
        cell_info[cid] = {
            'bbox': {0: (x, y, w, h)},
            'class': cls_id,
            'latest_frame_index': 0,
            'ref_tensor': transform(cell_pil),  # 保存参考tensor用于Siamese比较
            'pair_score_ema': None,
            'above_streak': 0,
            'state_history': {0: sickle_label},
            'aspect_ratio': {0: ar_val},
            'eccentricity': {0: ecc_val},
            'circularity': {0: circ_val}  # 需求4
        }

        frame0_records.append({
            'Cell_ID': cid,
            'Frame_Index': 0,
            'Aspect_Ratio': ar_val,
            'Eccentricity': ecc_val,
            'Circularity': circ_val,  # 需求4
            'Class_ID': cls_id,
            'Sickle_Label': sickle_label
        })

    frame0_df = pd.DataFrame(frame0_records)

    # Frame 0 是目标帧，输出统计
    if not frame0_df.empty and 0 in target_frames_set:
        all_frames_df = pd.concat([all_frames_df, frame0_df], ignore_index=True)
        frame0_df.to_csv(out_path + '/frame0_raw_data.csv', index=False)

        stats_ar = calculate_statistics_summary(frame0_df, ['Sickle_Label'], 'Aspect_Ratio')
        stats_ar.to_csv(out_path + '/frame0_stats_ar.csv', index=False)

        stats_ecc = calculate_statistics_summary(frame0_df, ['Sickle_Label'], 'Eccentricity')
        stats_ecc.to_csv(out_path + '/frame0_stats_ecc.csv', index=False)

        stats_circ = calculate_statistics_summary(frame0_df, ['Sickle_Label'], 'Circularity')  # 需求4
        stats_circ.to_csv(out_path + '/frame0_stats_circ.csv', index=False)

        plot_overall_nature_style_ar(frame0_df.copy(), out_path + '/frame0_violin_overall_ar.png', 0)
        plot_overall_nature_style_ar(frame0_df.copy(), out_path + '/frame0_violin_overall_ar_excl_G.png', 0,
                                     exclude_G=True)
        plot_7class_nature_style_ar(frame0_df.copy(), out_path + '/frame0_violin_7class_ar.png', 0)
        plot_overall_nature_style_ecc(frame0_df.copy(), out_path + '/frame0_violin_overall_ecc.png', 0)
        plot_overall_nature_style_ecc(frame0_df.copy(), out_path + '/frame0_violin_overall_ecc_excl_G.png', 0,
                                      exclude_G=True)
        plot_7class_nature_style_ecc(frame0_df.copy(), out_path + '/frame0_violin_7class_ecc.png', 0)

        # 需求4: Circularity绘图
        plot_overall_nature_style_circ(frame0_df.copy(), out_path + '/frame0_violin_overall_circ.png', 0)
        plot_overall_nature_style_circ(frame0_df.copy(), out_path + '/frame0_violin_overall_circ_excl_G.png', 0,
                                       exclude_G=True)

        plot_7class_nature_style_circ(frame0_df.copy(), out_path + '/frame0_violin_7class_circ.png', 0)

    # ========== 逐帧处理 Frame 1 到 max_target_frame (与原脚本一致) ==========
    print(f'\n- Processing frames 1 to {max_target_frame} (accumulating EMA/streak)...')

    for frame_idx in tqdm(range(1, max_target_frame + 1), desc='Processing frames'):
        ret, frame = cap.read()
        if not ret:
            print(f"  Warning: Cannot read frame {frame_idx}")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 是否是目标帧（需要输出统计）
        is_target_frame = (frame_idx in target_frames_set)

        if is_target_frame:
            cv2.imwrite(out_path + f"/frame_{frame_idx}.png", frame)

        # 分割当前帧
        bboxes, masks, unique_ids, _ = segment_frame_downscaled_ds(
            frame_rgb, cellpose_model_path, out_path,
            ratio=0.1, diameter=30, save_mask=False, frame_idx=frame_idx
        )

        # 匹配追踪细胞
        matched, used_cids = match_cells_tracking(cell_info, masks, bboxes)

        frame_records = []

        for cid, info in matched.items():
            x, y, w, h = info['bbox']
            cls_id = info['class']

            cell_crop = frame_rgb[y:y + h, x:x + w]
            if cell_crop.size == 0:
                continue

            cell_pil = Image.fromarray(cell_crop)

            with torch.no_grad():
                # 使用Siamese模型比较Frame 0的ref_tensor和当前帧
                ref_t = cell_info[cid]['ref_tensor'].to(device)
                cur_t = transform(cell_pil).to(device)

                logit = pair_model_All(ref_t.unsqueeze(0), cur_t.unsqueeze(0))
                p_changed = torch.sigmoid(logit[0]).item()

                # EMA平滑 (与原脚本一致)
                prev = cell_info[cid]['pair_score_ema']
                s_ema = p_changed if prev is None else (EMA_coeff * p_changed + (1 - EMA_coeff) * prev)
                cell_info[cid]['pair_score_ema'] = s_ema

                # 更新streak (与原脚本一致)
                is_above = (s_ema >= thr)
                streak = cell_info[cid].get('above_streak', 0)
                streak = streak + 1 if is_above else 0
                cell_info[cid]['above_streak'] = streak

                # 判断sickle label (与原脚本一致)
                # 需要连续 MIN_PERSIST 帧检测到变化才标记为 Sickle
                sickle_label = LABEL_SICKLE if (streak >= MIN_PERSIST) else LABEL_NONSICKLE

            # 更新cell_info
            cell_info[cid]['bbox'][frame_idx] = (x, y, w, h)
            cell_info[cid]['latest_frame_index'] = frame_idx
            cell_info[cid]['state_history'][frame_idx] = sickle_label

            # 只在目标帧计算AR/ECC并记录
            #if is_target_frame:
            is_dense_frame = (frame_idx % 2 == 0)
            if is_target_frame or is_dense_frame:
                # 计算当前帧的AR和ECC
                ar_val = 0.0
                ecc_val = 0.0
                circ_val = 0.0  # ← 移到这里初始化
                
                for mask_cid in unique_ids:
                    if mask_cid in bboxes:
                        mx, my, mw, mh = bboxes[mask_cid]
                        if abs(mx - x) < 50 and abs(my - y) < 50:
                            mask = (masks == mask_cid).astype(np.uint8)
                            ar_val = aspect_ratio(mask)
                            ecc_val = eccentricity(mask)
                            circ_val = circularity(mask)  # ← 加这行！
                            break

                # 如果没找到匹配的mask，使用bbox估算
                if ar_val == 0.0:
                    ar_val = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
                    ecc_val = 0.5
                    circ_val = 0.5

                cell_info[cid]['aspect_ratio'][frame_idx] = ar_val
                cell_info[cid]['eccentricity'][frame_idx] = ecc_val
                cell_info[cid]['circularity'][frame_idx] = circ_val  # 需求4

                frame_records.append({
                    'Cell_ID': cid,
                    'Frame_Index': frame_idx,
                    'Aspect_Ratio': ar_val,
                    'Eccentricity': ecc_val,
                    'Circularity': circ_val,  # 需求4
                    'Class_ID': cls_id,
                    'Sickle_Label': sickle_label
                })



        # 只在目标帧输出统计和图表
        if (is_target_frame or is_dense_frame) and frame_records:
            frame_df = pd.DataFrame(frame_records)
            all_frames_df = pd.concat([all_frames_df, frame_df], ignore_index=True)

            if not frame_df.empty:


                stats_ar = calculate_statistics_summary(frame_df, ['Sickle_Label'], 'Aspect_Ratio')


                stats_ecc = calculate_statistics_summary(frame_df, ['Sickle_Label'], 'Eccentricity')


                stats_circ = calculate_statistics_summary(frame_df, ['Sickle_Label'], 'Circularity')  # 需求4

                if frame_idx in save_image_frames:
                    frame_df.to_csv(out_path + f'/frame{frame_idx}_raw_data.csv', index=False)
                    stats_ar.to_csv(out_path + f'/frame{frame_idx}_stats_ar.csv', index=False)
                    stats_ecc.to_csv(out_path + f'/frame{frame_idx}_stats_ecc.csv', index=False)
                    stats_circ.to_csv(out_path + f'/frame{frame_idx}_stats_circ.csv', index=False)

                    plot_overall_nature_style_ar(frame_df.copy(), out_path + f'/frame{frame_idx}_violin_overall_ar.png',
                                                 frame_idx)
                    plot_7class_nature_style_ar(frame_df.copy(), out_path + f'/frame{frame_idx}_violin_7class_ar.png',
                                                frame_idx)
                    plot_overall_nature_style_ecc(frame_df.copy(), out_path + f'/frame{frame_idx}_violin_overall_ecc.png',
                                                  frame_idx)
                    plot_7class_nature_style_ecc(frame_df.copy(), out_path + f'/frame{frame_idx}_violin_7class_ecc.png',
                                                 frame_idx)

                    # 需求2: 增加排除G的overall图
                    plot_overall_nature_style_ar(frame_df.copy(),
                                                 out_path + f'/frame{frame_idx}_violin_overall_ar_excl_G.png',
                                                 frame_idx, exclude_G=True)
                    plot_overall_nature_style_ecc(frame_df.copy(),
                                                  out_path + f'/frame{frame_idx}_violin_overall_ecc_excl_G.png',
                                                  frame_idx, exclude_G=True)

                    # 需求4: Circularity绘图
                    plot_overall_nature_style_circ(frame_df.copy(), out_path + f'/frame{frame_idx}_violin_overall_circ.png',
                                                   frame_idx)
                    plot_overall_nature_style_circ(frame_df.copy(),
                                                   out_path + f'/frame{frame_idx}_violin_overall_circ_excl_G.png',
                                                   frame_idx, exclude_G=True)

                    plot_7class_nature_style_circ(frame_df.copy(), out_path + f'/frame{frame_idx}_violin_7class_circ.png',
                                                  frame_idx)

                    print(
                        f'  Frame {frame_idx}: Sickle={len(frame_df[frame_df["Sickle_Label"] == LABEL_SICKLE])}, Non-sickle={len(frame_df[frame_df["Sickle_Label"] == LABEL_NONSICKLE])}')
                else:
                    print(f"  Skipped plots for frame {frame_idx}")

    cap.release()

    if all_frames_df.empty:
        return all_frames_df

    # === 多帧汇总 ===
    print('\n- Generating multi-frame comparison plots...')

    # 保存汇总数据
    all_frames_df.to_csv(out_path + '/all_frames_raw_data.csv', index=False)

    # 汇总统计
    stats_all_ar = calculate_statistics_summary(all_frames_df, ['Frame_Index', 'Sickle_Label'], 'Aspect_Ratio')
    stats_all_ar.to_csv(out_path + '/all_frames_stats_ar.csv', index=False)

    stats_all_ecc = calculate_statistics_summary(all_frames_df, ['Frame_Index', 'Sickle_Label'], 'Eccentricity')
    stats_all_ecc.to_csv(out_path + '/all_frames_stats_ecc.csv', index=False)

    stats_all_circ = calculate_statistics_summary(all_frames_df, ['Frame_Index', 'Sickle_Label'], 'Circularity')  # 需求4
    stats_all_circ.to_csv(out_path + '/all_frames_stats_circ.csv', index=False)

    # 多帧对比图
    plot_multiframe_comparison_ar(all_frames_df, out_path + '/multiframe_comparison_ar.png', valid_target_frames)
    plot_multiframe_comparison_ecc(all_frames_df, out_path + '/multiframe_comparison_ecc.png', valid_target_frames)

    plot_multiframe_comparison_circ(all_frames_df, out_path + '/multiframe_comparison_circ.png',
                                    valid_target_frames)  # 需求4

    # 趋势图
    trend_stats = plot_multiframe_trend(all_frames_df, out_path + '/multiframe_trend.png', valid_target_frames,
                                        max_target_frame)
    if trend_stats is not None:
        trend_stats.to_csv(out_path + '/multiframe_trend_stats.csv', index=False)

    print(f"Done with {video_path}.")
    return all_frames_df


# -------------------- Main Execution ---------------------

print("-------------------- Parameterization --------------------")
parser = argparse.ArgumentParser(description='Multi-frame Sickle Cell AR & Eccentricity Analysis.')
parser.add_argument('-i', '--inputs', type=str, required=True,
                    help='Comma-separated list of input video files')
parser.add_argument('-o', '--output_dir', type=str, required=True,
                    help='Output directory')
parser.add_argument('--target_frames', type=str, default='0,480',
                    help='Comma-separated list of frame indices to analyze (default: 0,480)')
parser.add_argument('--frame_skip', type=int, default=2,
                    help='(Legacy parameter, not used in multi-frame mode)')
parser.add_argument('--max_frame', type=int, default=480,
                    help='(Legacy parameter, not used in multi-frame mode)')

args = parser.parse_args()

# 解析目标帧
target_frames = [int(f.strip()) for f in args.target_frames.split(',')]
print(f"Target frames to analyze: {target_frames}")

video_paths = args.inputs.split(',')
all_out = args.output_dir
out_path = [os.path.join(all_out, os.path.splitext(os.path.basename(v))[0]) for v in video_paths]

os.makedirs(all_out, exist_ok=True)
for op in out_path:
    os.makedirs(op, exist_ok=True)

all_videos_df = pd.DataFrame()

for idx, video_path in enumerate(video_paths):
    os.makedirs(out_path[idx], exist_ok=True)

    video_df = process_video_multiframe(
        video_path=video_path,
        out_path=out_path[idx],
        transform=transform,
        target_frames=target_frames
    )

    if not video_df.empty:
        video_df['Video_ID'] = f"V{idx + 1}"
        all_videos_df = pd.concat([all_videos_df, video_df], ignore_index=True)

print("\n" + "=" * 60)
print("Generating Combined Reports (All Videos)...")
print("=" * 60)

if not all_videos_df.empty:
    all_videos_df.to_csv(all_out + '/combined_all_videos_raw_data.csv', index=False)

    # Combined AR Stats
    stats_comb_ar = calculate_statistics_summary(all_videos_df, ['Frame_Index', 'Sickle_Label'], 'Aspect_Ratio')
    stats_comb_ar.to_csv(all_out + '/combined_stats_ar.csv', index=False)

    # Combined ECC Stats
    stats_comb_ecc = calculate_statistics_summary(all_videos_df, ['Frame_Index', 'Sickle_Label'], 'Eccentricity')
    stats_comb_ecc.to_csv(all_out + '/combined_stats_ecc.csv', index=False)

    # Combined Circularity Stats (需求4)
    stats_comb_circ = calculate_statistics_summary(all_videos_df, ['Frame_Index', 'Sickle_Label'], 'Circularity')
    stats_comb_circ.to_csv(all_out + '/combined_stats_circ.csv', index=False)
    save_image_frames = {480}

    # ====== 新增: Combined 每帧的 violin 图 ======
    print("\nGenerating combined per-frame violin plots...")
    for frame_idx in target_frames:
        if frame_idx not in save_image_frames:
            continue
        frame_df = all_videos_df[all_videos_df['Frame_Index'] == frame_idx]
        if not frame_df.empty:
            # Overall violin plots
            plot_overall_nature_style_ar(
                frame_df.copy(),
                all_out + f'/combined_frame{frame_idx}_violin_overall_ar.png',
                frame_idx
            )
            plot_overall_nature_style_ecc(
                frame_df.copy(),
                all_out + f'/combined_frame{frame_idx}_violin_overall_ecc.png',
                frame_idx
            )
            # 7-class violin plots
            plot_7class_nature_style_ar(
                frame_df.copy(),
                all_out + f'/combined_frame{frame_idx}_violin_7class_ar.png',
                frame_idx
            )
            plot_7class_nature_style_ecc(
                frame_df.copy(),
                all_out + f'/combined_frame{frame_idx}_violin_7class_ecc.png',
                frame_idx
            )

            # 需求2 & 4: 添加exclude_G版本和circularity
            # Overall plots - exclude G
            plot_overall_nature_style_ar(
                frame_df.copy(),
                all_out + f'/combined_frame{frame_idx}_violin_overall_ar_excl_G.png',
                frame_idx,
                exclude_G=True
            )
            plot_overall_nature_style_ecc(
                frame_df.copy(),
                all_out + f'/combined_frame{frame_idx}_violin_overall_ecc_excl_G.png',
                frame_idx,
                exclude_G=True
            )

            # Circularity plots
            plot_overall_nature_style_circ(
                frame_df.copy(),
                all_out + f'/combined_frame{frame_idx}_violin_overall_circ.png',
                frame_idx
            )
            plot_overall_nature_style_circ(
                frame_df.copy(),
                all_out + f'/combined_frame{frame_idx}_violin_overall_circ_excl_G.png',
                frame_idx,
                exclude_G=True
            )
            plot_7class_nature_style_circ(
                frame_df.copy(),
                all_out + f'/combined_frame{frame_idx}_violin_7class_circ.png',
                frame_idx
            )
            print(f"  - Generated combined violin plots for frame {frame_idx}")

    # Combined Plots (multi-frame comparison)
    plot_multiframe_comparison_ar(all_videos_df, all_out + '/combined_multiframe_comparison_ar.png', target_frames)
    plot_multiframe_comparison_ecc(all_videos_df, all_out + '/combined_multiframe_comparison_ecc.png', target_frames)

    plot_multiframe_comparison_circ(all_videos_df, all_out + '/combined_multiframe_comparison_circ.png',
                                    target_frames)  # 需求4

    trend_stats = plot_multiframe_trend(all_videos_df, all_out + '/combined_multiframe_trend.png', target_frames,
                                        max(target_frames))
    if trend_stats is not None:
        trend_stats.to_csv(all_out + '/combined_multiframe_trend_stats.csv', index=False)

    print("\n" + "=" * 60)
    print("========== 处理完成 ==========")
    print("=" * 60)
    print(f"\n结果保存在 {all_out}:")
    print("\n  [每个视频单独目录]")
    print("    - frameX_raw_data.csv          (每帧原始数据)")
    print("    - frameX_stats_ar/ecc.csv      (每帧统计)")
    print("    - frameX_violin_*.png          (每帧小提琴图)")
    print("    - all_frames_raw_data.csv      (所有帧汇总)")
    print("    - multiframe_comparison_*.png  (多帧对比图)")
    print("    - multiframe_trend.png         (趋势图)")
    print("\n  [汇总目录 - Combined]")
    print("    - combined_all_videos_raw_data.csv")
    print("    - combined_stats_ar/ecc.csv")
    print("    - combined_frameX_violin_overall_ar.png   (新增: 每帧Overall AR小提琴图)")
    print("    - combined_frameX_violin_overall_ecc.png  (新增: 每帧Overall ECC小提琴图)")
    print("    - combined_frameX_violin_7class_ar.png    (新增: 每帧7-class AR小提琴图)")
    print("    - combined_frameX_violin_7class_ecc.png   (新增: 每帧7-class ECC小提琴图)")
    print("    - combined_multiframe_*.png")
    print(f"\n分析的帧: {target_frames}")
else:
    print("Warning: No cell data collected from any video.")
