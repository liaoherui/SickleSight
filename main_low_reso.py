# Runn command:
# python main.py -i /data/videos          # 默认 120s
# python main.py -i /data/videos --max-time 60   # 改成 60s
# python main.py -i /data/videos --max-frame 400  # 回到帧数模式


import os
from turtle import home
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps
from torchvision import transforms, models
from ultralytics import YOLO
from scipy.signal import savgol_filter
from collections import deque, defaultdict, Counter
import datetime
import sys

from device_utils import get_torch_device, get_ultralytics_device

# ===== Numpy compatibility fix (numpy 2.x -> 1.x) =====
# Fix: numpy._core module missing in older numpy versions
if not hasattr(np, '_core'):
    np._core = np.core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core.multiarray'] = np.core.multiarray

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CELLBOX_MODELS_DIR = os.path.join(SCRIPT_DIR, 'CellBox-Models')

#from ultralytics.models import yolo

# ======================================
# Hyperparameters Configuration
# ======================================

class Config:
    """Configuration class containing all hyperparameters"""
    

    # ========== Time window parameters ==========
    # Time window size in SECONDS per cell class (tuned at 4 fps; auto-converted to frames at runtime)
    CLASS_TIME_WINDOWS_SEC = {
        'Dis': 2.75,   # 11 frames @ 4fps
        'Cup': 2.75,
        'Sto': 2.75,
        'Ret': 8.75,   # 35 frames @ 4fps
        'Ech': 2.75,
        'Gra': 2.75,
        'ISC': 2.75,
    }

    @classmethod
    def get_time_windows_frames(cls, fps):
        """Convert CLASS_TIME_WINDOWS_SEC to frames for a given fps. Result is always odd."""
        windows = {}
        for c, sec in cls.CLASS_TIME_WINDOWS_SEC.items():
            frames = max(3, int(round(sec * fps)))
            if frames % 2 == 0:
                frames += 1  # keep odd so half-window is symmetric
            windows[c] = frames
        return windows

    # Other time-related parameters
    MAX_VIDEO_FRAMES = 480  # fallback when MAX_TIME_S is None
    MAX_TIME_S = 120.0      # default: process first 120s per video (converted to frames at runtime)
    
    # ========== Processing parameters ==========
    MIN_CELL_AREA        = 30000    # Minimum cell bounding box area at REF_RESOLUTION
    REF_RESOLUTION       = (5472, 3648)  # resolution where MIN_CELL_AREA was tuned
    MARGIN               = 3      # Edge margin for truncated cell filtering (pixels)
    EDGE_BOX_AR_MAX      = 1.4    # Max aspect ratio for edge boxes before discarding

    # Confidence thresholds
    DETECTION_CONFIDENCE = 0.25   # YOLO detection confidence
    CLASSIFICATION_CONFIDENCE = 0.8  # Classification confidence
    YOLO_IOU = 0.6                # YOLO tracking IOU threshold

    # Which prediction stage to visualize: raw / ret / smoothed / final
    VISUALIZATION_SICKLE_SOURCE = 'final'

    # Whether to enable Ret-specific constraints
    USE_RET_CONSTRAINT = False

    # Whether to enable seg model for Dis/ISC aspect-ratio reclassification
    USE_SEG_RECLASSIFY = True
    SEG_ASPECT_RATIO_THRESHOLD = 1.6  # ISC aspect ratio threshold (major/minor > this -> ISC)

    # Auto detection-confidence estimation
    AUTO_DET_CONF = True  # if True, probe video before processing to set DETECTION_CONFIDENCE

    # Initial cell counting parameters
    INITIAL_FRAMES_COUNT = 1  # Collect unique cells from first N frames as initial count
    
    # Bounding box filter parameters
    MARGIN = 3           # Edge filter margin (pixels)
    EDGE_BOX_AR_MAX = 1.4  # Edge box AR threshold: discard if exceeded (truncated cells)

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("Hyperparameters Configuration")
        print("=" * 60)
        print(f"  Video count: {len(cls.VIDEO_PATHS)}")
        for i, path in enumerate(cls.VIDEO_PATHS, 1):
            print(f"   Video {i}: {Path(path).name}")
        print("  Time window config (seconds, fps-adaptive):")
        for cell_type, sec in cls.CLASS_TIME_WINDOWS_SEC.items():
            print(f"    {cell_type}: {sec}s")
        print(f"  Max frames: {cls.MAX_VIDEO_FRAMES} frames/video")
        print(f"  Detection confidence: {cls.DETECTION_CONFIDENCE}")
        print(f"  Classification confidence: {cls.CLASSIFICATION_CONFIDENCE}")
        print(f"  Tracking IOU: {cls.YOLO_IOU}")
        print(f"  Visualization sickle source: {cls.VISUALIZATION_SICKLE_SOURCE}")
        print(f"  Edge margin: {cls.MARGIN} px")
        print(f"  Edge box AR threshold: {cls.EDGE_BOX_AR_MAX} (discard if exceeded)")
        print("=" * 60)

# Siamese preprocessing transforms
siamese_img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t * 2.0 - 1.0)  # linear map to [-1,1]
])
siamese_mask_transform = transforms.ToTensor()

# ======================================
# Global class configuration
# ======================================
MC_CLASSES = ['Dis', 'Cup', 'Sto', 'Ret', 'Ech', 'Gra', 'ISC']  # A_Disocyte, B_Cupped, C_Stomatocyte, D_Reticulocyte, E_Echino, F_Granular, G_ISCs
CLASS_LETTERS = {'Dis': 'A', 'Cup': 'B', 'Sto': 'C', 'Ret': 'D', 'Ech': 'E', 'Gra': 'F', 'ISC': 'G'}
SIAMESE_CLASSES = ['Dis', 'Cup', 'Sto', 'Ret', 'Ech', 'Gra', 'ISC']

CURVE_COLORS = {
    'A': '#0922e3',   # Dis - blue
    'B': '#099AE3',   # Cup - sky blue
    'C': '#E39E09',   # Sto - golden yellow
    'D': '#9E09E3',   # Ret - purple
    'E': '#895129',   # Ech - brown
    'F': '#09E360',   # Gra - green
    'G': '#F05151',   # ISC - red
}
COLOR_MAP = {c: CURVE_COLORS[CLASS_LETTERS[c]] for c in MC_CLASSES}

COLORS = {
    'Dis': (139, 0, 0),      # dark red
    'Cup': (255, 255, 0),    # yellow
    'Sto': (0, 0, 255),      # blue
    'Ret': (0, 255, 0),      # green
    'Ech': (0, 140, 255),    # orange-blue
    'Gra': (255, 165, 0),    # orange
    'ISC': (128, 0, 128)     # purple
}
DEO_COLOR = (0, 255, 255)

def display_name(c, count=None):
    """Generate display name for a class"""
    letter = CLASS_LETTERS.get(c, '?')
    if count is not None:
        return f"{letter}_{c}({count})"
    else:
        return f"{letter}_{c}"

def reclassify_by_aspect_ratio(crop, seg_model, base_label, conf_threshold=0.05,
                               aspect_ratio_threshold=1.7, yolo_device=None):
    """
    Reclassify Dis and ISC labels using seg model aspect ratio.
    
    Args:
        crop: Cell crop image (BGR)
        seg_model: YOLOv8 segmentation model
        base_label: Initial label
        conf_threshold: Seg model confidence threshold
        aspect_ratio_threshold: Aspect ratio threshold for ISC
    
    Returns:
        reclassified_label: Updated label
        aspect_ratio: Computed ratio, or None
    """
    # Only reclassify Dis and ISC cells
    if base_label not in ['Dis', 'ISC']:
        return base_label, None
    
    try:
        # Run seg model prediction
        results = seg_model.predict(
            source=crop,
            conf=conf_threshold,
            save=False,
            verbose=False,
            device=yolo_device,
        )
        
        if results[0].masks is None or len(results[0].masks.data) == 0:
            # No mask detected — keep original label
            return base_label, None
        
        # Use first mask (highest confidence)
        masks = results[0].masks.data.cpu().numpy()
        mask = masks[0]

        # Resize mask to crop dimensions
        h, w = crop.shape[:2]
        mask_resized = cv2.resize(mask, (w, h))
        binary = (mask_resized > 0.5).astype(np.uint8)

        # Compute aspect ratio via ellipse fitting
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return base_label, None

        largest_contour = max(contours, key=cv2.contourArea)
        
        # Use ellipse major/minor axes
        if len(largest_contour) >= 5:  # ellipse fitting requires >=5 points
            ellipse = cv2.fitEllipse(largest_contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes)  # major axis
            minor_axis = min(axes)  # minor axis
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
        else:
            # Too few points: fall back to minimum bounding rect
            rect = cv2.minAreaRect(largest_contour)
            width, height = rect[1]
            major_axis = max(width, height)
            minor_axis = min(width, height)
            aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0
        
        # Reclassify based on aspect ratio (bidirectional)
        if aspect_ratio > aspect_ratio_threshold:
            reclassified_label = 'ISC'
        else:
            reclassified_label = 'Dis'

        return reclassified_label, aspect_ratio
        
    except Exception as e:
        print(f"    Warning: Aspect ratio reclassification failed: {e}")
        return base_label, None

# Common utility functions

def get_model_and_transform(model_name, model_path, num_classes, device):
    """Load classification model and corresponding preprocessing transform"""
    if model_name == "ResNet18":
        model = models.resnet18(weights=None)
        input_size = (224, 224)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "EfficientNet_b3":
        model = models.efficientnet_b3(weights=None)
        input_size = (224, 224)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "ConvNeXt_Tiny":
        model = models.convnext_tiny(weights=None)
        input_size = (224, 224)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return model, transform



def filter_boxes(results, img_w, img_h, min_cell_area=None):
    """
    Filter YOLO detection boxes.
    - Interior cells (distance > MARGIN): always kept
    - Edge cells (distance <= MARGIN): discard if AR > EDGE_BOX_AR_MAX (truncated)
    min_cell_area: adaptive threshold (computed from video resolution); falls back to Config value.
    """
    if min_cell_area is None:
        min_cell_area = Config.MIN_CELL_AREA
    filtered, ids = [], []
    if not hasattr(results, 'boxes') or results.boxes is None:
        return filtered, ids
    boxes = results.boxes
    ids_raw = (boxes.id.cpu().numpy()
               if (boxes.id is not None and len(boxes.id) > 0)
               else np.arange(len(boxes.xyxy)))
    for box, tid in zip(boxes.xyxy, ids_raw):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        w, h = x2 - x1, y2 - y1
        if w * h < min_cell_area:
            continue
        at_edge = (x1 < Config.MARGIN or y1 < Config.MARGIN or
                   x2 > img_w - Config.MARGIN or y2 > img_h - Config.MARGIN)
        if at_edge and max(w, h) / max(min(w, h), 1) > Config.EDGE_BOX_AR_MAX:
            continue
        filtered.append((x1, y1, x2, y2))
        ids.append(int(tid))

    if len(filtered) == 0:
        return filtered, ids

    # Post-NMS: remove remaining overlapping boxes (IoU > 0.4), keep larger box
    import torchvision
    boxes_t = torch.tensor(filtered, dtype=torch.float32)
    areas   = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
    keep    = torchvision.ops.nms(boxes_t, areas, iou_threshold=0.4).tolist()
    filtered = [filtered[i] for i in keep]
    ids      = [ids[i]      for i in keep]
    return filtered, ids

def pad_resize_gray(img: Image.Image, target_size=(224,224), fill=(128,128,128)):
    """Pad and resize image with letterboxing"""
    w,h = img.size
    m = max(w,h)
    pad = ((m-w)//2, (m-h)//2, (m-w+1)//2, (m-h+1)//2)
    img = ImageOps.expand(img, pad, fill=fill)
    return img.resize(target_size, Image.BILINEAR)

# Siamese binary classification model

import torch.nn as nn

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation Layer"""
    def __init__(self, feature_dim, num_cell_types=7, embed_dim=128):
        super(FiLMLayer, self).__init__()
        self.type_embed = nn.Embedding(num_cell_types, embed_dim)
        self.gamma_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, feature_dim)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, feature_dim)
        )
        with torch.no_grad():
            self.gamma_net[-1].weight.fill_(0.1)
            self.gamma_net[-1].bias.fill_(1.0)
            self.beta_net[-1].weight.fill_(0.1)
            self.beta_net[-1].bias.fill_(0.0)
    
    def forward(self, features, cell_type):
        batch_size = features.size(0)
        type_vec = self.type_embed(cell_type)
        gamma = self.gamma_net(type_vec)
        beta = self.beta_net(type_vec)
        gamma = gamma.view(batch_size, -1, 1, 1)
        beta = beta.view(batch_size, -1, 1, 1)
        return gamma * features + beta

class ChannelAttention(nn.Module):
    """Channel attention mechanism"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ConditionalFeatureExtractor(nn.Module):
    """Conditional feature extractor with FiLM and CBAM using full ResNet18 architecture"""
    def __init__(self, num_cell_types=7, use_film=True, use_cbam=True):
        super(ConditionalFeatureExtractor, self).__init__()
        self.use_film = use_film
        self.use_cbam = use_cbam
        
        # Use pretrained ResNet18
        from torchvision.models import resnet18
        try:
            from torchvision.models import ResNet18_Weights
            resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except:
            resnet = resnet18(pretrained=True)
        
        # Extract ResNet18 layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels
        
        self.avgpool = resnet.avgpool
        
        # FiLM layer (applied at last layer only)
        if self.use_film:
            self.film_layer4 = FiLMLayer(512, num_cell_types)
        
        # CBAM layers
        if self.use_cbam:
            self.cbam_layer1 = CBAM(64)
            self.cbam_layer2 = CBAM(128)
            self.cbam_layer3 = CBAM(256)
            self.cbam_layer4 = CBAM(512)
    
    def forward(self, x, cell_type):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Layer 1
        x = self.layer1(x)
        if self.use_cbam:
            x = self.cbam_layer1(x)
        
        # Layer 2
        x = self.layer2(x)
        if self.use_cbam:
            x = self.cbam_layer2(x)
        
        # Layer 3
        x = self.layer3(x)
        if self.use_cbam:
            x = self.cbam_layer3(x)
        
        # Layer 4 (final)
        x = self.layer4(x)
        # Apply FiLM only at final layer
        if self.use_film:
            x = self.film_layer4(x, cell_type)
        if self.use_cbam:
            x = self.cbam_layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

class ConditionalSiameseNetwork(nn.Module):
    """Conditional Siamese network for sickle cell detection (matches training architecture)"""
    def __init__(self, num_cell_types=7, use_film=True, use_cbam=True, fusion_mode='B'):
        super(ConditionalSiameseNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = ConditionalFeatureExtractor(
            num_cell_types=num_cell_types, 
            use_film=use_film, 
            use_cbam=use_cbam
        )
        
        # Feature fusion mode:
        #   A: [f1, f2]
        #   B: [f1, f2, cosine]  (default)
        #   C: [|f1-f2|, f1*f2]
        assert fusion_mode in ['A', 'B', 'C'], f"Unsupported fusion_mode: {fusion_mode}"
        self.fusion_mode = fusion_mode
        
        # Classifier head (input dim depends on fusion mode)
        if self.fusion_mode == 'A':
            classifier_in_dim = 512 * 2  # [f1, f2]
        elif self.fusion_mode == 'B':
            classifier_in_dim = 512 * 2 + 1  # [f1, f2, cosine]
        else:  # 'C'
            classifier_in_dim = 512 * 2  # [|f1-f2|, f1*f2]

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # binary output
        )
    
    def forward(self, img1, img2, cell_type):
        # Extract features from both images
        feat1 = self.feature_extractor(img1, cell_type)
        feat2 = self.feature_extractor(img2, cell_type)
        
        # Build fused feature vector
        if self.fusion_mode == 'A':
            combined_features = torch.cat([feat1, feat2], dim=1)
        elif self.fusion_mode == 'B':
            cosine_sim = F.cosine_similarity(feat1, feat2, dim=1, eps=1e-8).unsqueeze(1)
            combined_features = torch.cat([feat1, feat2, cosine_sim], dim=1)
        else:  # 'C'
            diff = torch.abs(feat1 - feat2)
            prod = feat1 * feat2
            combined_features = torch.cat([diff, prod], dim=1)
        
        # Classify
        similarity = self.classifier(combined_features)
        return similarity

# Time window smoothing
# (original single-class version, kept for reference)
# def apply_time_window_smoothing(sickle_predictions, window_size=5):
#     """
#     Apply time-window smoothing per track_id.
#     A frame is sickle only if ALL frames in the window are sickle.
#     Once sickle, all subsequent frames stay sickle.
#     """
#     smoothed_predictions = {}
#     half_window = window_size // 2
#     
#     for track_id, predictions in sickle_predictions.items():
#         smoothed = []
#         sickle_locked = False
#         
#         for i in range(len(predictions)):
#             if sickle_locked:
#                 smoothed_result = True
#             else:
#                 start_idx = max(0, i - half_window)
#                 end_idx = min(len(predictions), i + half_window + 1)
#                 window_preds = predictions[start_idx:end_idx]
#                 smoothed_result = all(window_preds)
#                 
#                 if smoothed_result:
#                     sickle_locked = True
#             
#             smoothed.append(smoothed_result)
#         
#         smoothed_predictions[track_id] = smoothed
#     
#     return smoothed_predictions

def apply_time_window_smoothing(sickle_predictions, initial_labels=None, time_windows_frames=None):
    """
    Apply time-window smoothing per track_id.
    A frame is sickle only if ALL frames in the window are sickle.
    Once labeled sickle, all subsequent frames remain sickle.
    time_windows_frames: per-class frame counts from Config.get_time_windows_frames(fps).
    """
    smoothed_predictions = {}
    if time_windows_frames is None:
        time_windows_frames = Config.get_time_windows_frames(4.0)  # fallback: assume 4 fps

    for track_id, predictions in sickle_predictions.items():
        # Look up cell class for this track
        cell_label = None
        if initial_labels and track_id in initial_labels:
            cell_label = initial_labels[track_id]

        # Select window size based on cell class
        if cell_label and cell_label in time_windows_frames:
            current_window_size = time_windows_frames[cell_label]
        else:
            current_window_size = time_windows_frames.get('Dis', 11)  # fallback to Dis window
            if cell_label:
                print(f"Warning: Unknown cell type '{cell_label}' for track {track_id}, using default window size {current_window_size}")
        
        half_window = current_window_size // 2
        
        # Apply sliding-window smoothing
        smoothed = []
        sickle_locked = False
        
        for i in range(len(predictions)):
            if sickle_locked:
                smoothed_result = True
            else:
                start_idx = max(0, i - half_window)
                end_idx = min(len(predictions), i + half_window + 1)
                window_preds = predictions[start_idx:end_idx]
                smoothed_result = all(window_preds)
                
                if smoothed_result:
                    sickle_locked = True
            
            smoothed.append(smoothed_result)
        
        smoothed_predictions[track_id] = smoothed
    
    return smoothed_predictions

# Verify and fix sickle count monotonicity
def verify_and_fix_monotonicity(frame_stats, MC_CLASSES):
    """
    Verify and fix sickle count monotonicity.
    Ensures per-class sickle counts are non-decreasing and never exceed the total.
    """
    
    for c in MC_CLASSES:
        sickle_col = f"{c}_sickle"
        total_col = f"{c}_total"
        prev_sickle = 0
        
        fixed_count = 0
        for i, frame_stat in enumerate(frame_stats):
            current_sickle = frame_stat[sickle_col]
            total_count = frame_stat[total_col]
            
            # Fix 1: clamp up if count dropped below previous frame
            if current_sickle < prev_sickle:
                frame_stats[i][sickle_col] = prev_sickle
                current_sickle = prev_sickle
                fixed_count += 1
            
            # Fix 2: clamp down if count exceeds total
            if current_sickle > total_count and total_count > 0:
                frame_stats[i][sickle_col] = total_count
                current_sickle = total_count
                fixed_count += 1
            
            prev_sickle = current_sickle
        
        if fixed_count > 0:
            print(f"    Fixed {fixed_count} frames for {c} class sickle count")
    
    return frame_stats
def apply_ech_ret_similarity_constraint(sickle_predictions, all_frame_data, initial_images, initial_labels, device, siamese, fps):
    """
    Apply similarity constraint for Ech and Ret cells.
    If a cell shows sickle in the first 20 s and first-last similarity > 50%,
    mark all frames as non-sickle.
    """
    print("  Applying Ech and Ret similarity constraints...")

    constrained_predictions = {}

    # Preprocessing transform
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for track_id, predictions in sickle_predictions.items():
        # Only apply to Ech and Ret cells
        cell_label = initial_labels.get(track_id, '')
        if cell_label not in ['Ech', 'Ret']:
            constrained_predictions[track_id] = predictions.copy()
            continue

        constrained = predictions.copy()

        try:
            # Collect detections for this track
            track_detections = {}
            for frame_data in all_frame_data:
                frame_id = frame_data['frame_id']
                for detection in frame_data['detections']:
                    if detection['track_id'] == track_id:
                        track_detections[frame_id] = detection
                        break

            if len(track_detections) < 10:  # too few frames, skip
                constrained_predictions[track_id] = constrained
                continue

            # Check if sickle appeared in first 20 s
            early_sickle_detected = False
            early_frames_limit = int(20 * fps)
            
            for frame_id in range(min(early_frames_limit, len(predictions))):
                if frame_id < len(predictions) and predictions[frame_id]:
                    early_sickle_detected = True
                    break
            
            # No early sickle — skip constraint
            if not early_sickle_detected:
                constrained_predictions[track_id] = constrained
                continue

            # Get first and last frame IDs
            frame_ids = sorted(track_detections.keys())
            first_frame_id = frame_ids[0]
            last_frame_id = frame_ids[-1]

            # First frame (already preprocessed)
            proc0, _ = initial_images[track_id]
            first_tensor = transform_eval(proc0).unsqueeze(0).to(device)

            # Last frame
            last_detection = track_detections[last_frame_id]
            last_pil_rgb = Image.fromarray(cv2.cvtColor(last_detection['crop'], cv2.COLOR_BGR2RGB))
            #last_proc = pad_resize_gray(last_pil_rgb)
            last_tensor = transform_eval(last_pil_rgb).unsqueeze(0).to(device)

            # Compute first-last cosine similarity
            cell_type_idx = SIAMESE_CLASSES.index(cell_label)
            cell_type_tensor = torch.tensor([cell_type_idx], dtype=torch.long).to(device)
            
            with torch.no_grad():
                feat_first = siamese.feature_extractor(first_tensor, cell_type_tensor)
                feat_last = siamese.feature_extractor(last_tensor, cell_type_tensor)
                cosine_sim = F.cosine_similarity(feat_first, feat_last, dim=1).item()

            print(f"    Track {track_id} ({cell_label}): early sickle detected, first-last similarity = {cosine_sim:.4f}")

            # If similarity > 50%, mark all frames non-sickle
            if cosine_sim > 0.5:
                print(f"    Track {track_id} ({cell_label}): High similarity ({cosine_sim:.4f} > 0.5), setting all frames to non-sickle")
                constrained = [False] * len(predictions)
            else:
                print(f"    Track {track_id} ({cell_label}): Low similarity ({cosine_sim:.4f} <= 0.5), keeping original predictions")

        except Exception as e:
            print(f"    Warning: Failed to apply Ech/Ret similarity constraint for track {track_id}: {e}")
            # On error, keep original predictions
            constrained = predictions.copy()

        constrained_predictions[track_id] = constrained

    return constrained_predictions


def apply_ret_constraint(sickle_predictions, all_frame_data, initial_images, initial_labels, device, siamese, fps):
    """
    Apply Ret-specific constraint.
    1. Compute baseline cosine similarity (first vs last frame)
    2. Mid frames with similarity < baseline/2 are set to non-sickle
    3. Apply stricter time constraint for the early period
    """
    print("  Applying Ret-specific constraints...")

    constrained_predictions = {}

    # Preprocessing transform
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    ret_cell_type_idx = SIAMESE_CLASSES.index('Ret')

    for track_id, predictions in sickle_predictions.items():
        # Only apply to Ret cells
        cell_label = initial_labels.get(track_id, '')
        if cell_label != 'Ret':
            constrained_predictions[track_id] = predictions.copy()
            continue

        constrained = predictions.copy()

        try:
            # Collect detections for this track
            track_detections = {}
            for frame_data in all_frame_data:
                frame_id = frame_data['frame_id']
                for detection in frame_data['detections']:
                    if detection['track_id'] == track_id:
                        track_detections[frame_id] = detection
                        break

            if len(track_detections) < 10:  # too few frames, skip
                constrained_predictions[track_id] = constrained
                continue

            # 首尾帧
            frame_ids = sorted(track_detections.keys())
            first_frame_id = frame_ids[0]
            last_frame_id = frame_ids[-1]

            # First frame (already preprocessed)
            proc0, _ = initial_images[track_id]
            first_tensor = transform_eval(proc0).unsqueeze(0).to(device)

            # Last frame
            last_detection = track_detections[last_frame_id]
            last_pil_rgb = Image.fromarray(cv2.cvtColor(last_detection['crop'], cv2.COLOR_BGR2RGB))
            #last_proc = pad_resize_gray(last_pil_rgb)
            last_tensor = transform_eval(last_pil_rgb).unsqueeze(0).to(device)

            # Baseline similarity (first vs last frame)
            cell_type_tensor = torch.tensor([ret_cell_type_idx], dtype=torch.long).to(device)
            with torch.no_grad():
                feat_first = siamese.feature_extractor(first_tensor, cell_type_tensor)
                feat_last = siamese.feature_extractor(last_tensor, cell_type_tensor)
                baseline_cosine_sim = F.cosine_similarity(feat_first, feat_last, dim=1).item()

            print(f"    Track {track_id} (Ret): baseline cosine similarity (first vs last) = {baseline_cosine_sim:.4f}")

            # Mid-frame threshold: similarity < baseline/2 -> non-sickle
            similarity_threshold = baseline_cosine_sim * 0.5

            constrained_count = 0
            for frame_id in frame_ids[1:-1]:  # exclude first and last
                if frame_id in track_detections and constrained[frame_id]:
                    current_detection = track_detections[frame_id]
                    current_pil_rgb = Image.fromarray(cv2.cvtColor(current_detection['crop'], cv2.COLOR_BGR2RGB))
                    #current_proc = pad_resize_gray(current_pil_rgb)
                    current_tensor = transform_eval(current_pil_rgb).unsqueeze(0).to(device)

                    with torch.no_grad():
                        feat_current = siamese.feature_extractor(current_tensor, cell_type_tensor)
                        current_cosine_sim = F.cosine_similarity(feat_first, feat_current, dim=1).item()

                        if current_cosine_sim < similarity_threshold:
                            constrained[frame_id] = False
                            constrained_count += 1

            # Extra time-based constraint for early period
            early_constraint_frames = min(int(40 * fps), len(predictions))  # first 40 s
            early_constrained = 0

            for i in range(early_constraint_frames):
                if i < len(constrained) and constrained[i] and (i / fps) < 30:
                    time_factor = i / (fps * 30)
                    if baseline_cosine_sim > 0.8 and time_factor < 0.8:
                        constrained[i] = False
                        early_constrained += 1

            print(f"    Track {track_id} (Ret): constrained {constrained_count} frames by similarity, {early_constrained} frames by time")

        except Exception as e:
            print(f"    Warning: Failed to apply detailed Ret constraint for track {track_id}: {e}")
            # Fall back to simple time constraint
            early_frames = min(int(30 * fps), len(predictions))  # first 30 s
            for i in range(early_frames):
                if (i / fps) < 20.0:  # first 20 s
                    if i < len(constrained):
                        constrained[i] = False

        constrained_predictions[track_id] = constrained

    return constrained_predictions




def auto_detect_conf(video_path, yolo_model, n_frames=5, yolo_device=None):
    """Probe the video to automatically choose DETECTION_CONFIDENCE.

    Runs YOLO at conf=0.1 on a few early frames and measures the fraction of
    detections whose confidence score is >= 0.9 (high-confidence fraction).

    Sparse videos (cells well-separated) → YOLO is very confident → high frac → high conf.
    Dense videos (cells overlapping)     → many borderline detections → low frac → low conf.

    Calibrated on:
      V1.mp4   (sparse, optimal conf=0.80): high_frac ≈ 0.78
      MD11.mp4 (dense,  optimal conf=0.25): high_frac ≈ 0.58

    Thresholds:
      high_frac >= 0.72  → sparse  → conf 0.80
      high_frac >= 0.62  → medium  → conf 0.50
      high_frac <  0.62  → dense   → conf 0.25
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(5, min(total - 1, 100), n_frames, dtype=int)

    fracs = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        results = yolo_model.predict(
            frame,
            conf=0.1,
            max_det=2000,
            verbose=False,
            device=yolo_device,
        )[0]
        if results.boxes is not None and len(results.boxes) > 0:
            confs = results.boxes.conf.cpu().numpy()
            fracs.append(float(np.mean(confs >= 0.9)))
    cap.release()

    if not fracs:
        print(f"  [auto-conf] No detections in probe frames; keeping default "
              f"{Config.DETECTION_CONFIDENCE}")
        return Config.DETECTION_CONFIDENCE

    avg_frac = float(np.mean(fracs))

    if avg_frac >= 0.72:
        conf, density = 0.80, "sparse"
    elif avg_frac >= 0.62:
        conf, density = 0.50, "medium"
    else:
        conf, density = 0.25, "dense"

    print(f"  [auto-conf] probed {len(fracs)} frames | "
          f"high-conf-frac={avg_frac:.3f} | density={density} | conf={conf}")
    return conf


# ──────────────────────────────────────────────────────────────────────────────
# Per-video plot helpers  (match sicklesight output naming convention)
# ──────────────────────────────────────────────────────────────────────────────

def plot_state_ratio_binary(frame_stats, output_dir):
    """Overall (binary) sickle fraction over time → state_ratio_plot_binary.png"""
    times, ratios = [], []
    for f in frame_stats:
        total_all  = sum(f.get(f'{c}_total',  0) for c in MC_CLASSES)
        sickle_all = sum(f.get(f'{c}_sickle', 0) for c in MC_CLASSES)
        times.append(f['time_s'])
        ratios.append(sickle_all / total_all * 100 if total_all > 0 else 0.0)
    plt.figure(figsize=(8, 5))
    plt.plot(times, ratios, label='Total sickled fraction', color='black')
    plt.xlabel('Time (s)')
    plt.ylabel('Sickled fraction (%)')
    plt.ylim(0, 100)
    plt.title('Total cell ratio (binary)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(output_dir, 'state_ratio_plot_binary.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  ✓ state_ratio_plot_binary.png")
    csv_out = out.replace('.png', '.csv')
    pd.DataFrame({'Time_sec': times,
                  'Sickled_fraction_percent (%)': ratios}).to_csv(csv_out, index=False)


def plot_frame0_class_pie(label_counts, output_dir):
    """Pie chart of initial 7-class distribution → frame0_class_pie.png"""
    sizes = [label_counts.get(c, 0) for c in MC_CLASSES]
    total_cells = sum(sizes)
    if total_cells == 0:
        return
    colors7 = [COLOR_MAP[c] for c in MC_CLASSES]
    pcts = [s / total_cells * 100 for s in sizes]
    legend_labels = [
        f"{CLASS_LETTERS[c]}: {sizes[i]} ({pcts[i]:.1f}%)"
        for i, c in enumerate(MC_CLASSES)
    ]
    _, ax = plt.subplots(figsize=(8, 6))
    wedges, _ = ax.pie(sizes, startangle=140, colors=colors7,
                       wedgeprops=dict(width=0.5))
    ax.legend(wedges, legend_labels, title="Classes", loc="center left",
              bbox_to_anchor=(0.92, 0.5), fontsize=10)
    ax.set_title("Class Distribution in Frame 0", fontsize=14)
    plt.tight_layout()
    out = os.path.join(output_dir, 'frame0_class_pie.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  ✓ frame0_class_pie.png")


# Single-video processing
def process_single_video(video_path, models_dict, device, output_dir):
    """Process a single video; return per-frame statistics."""
    print(f"\nProcessing video: {os.path.basename(video_path)}")
    
    six2siamese = {name: SIAMESE_CLASSES.index(name) for name in MC_CLASSES if name in SIAMESE_CLASSES}
    
    yolo = models_dict['yolo']
    seg_model = models_dict['seg_model']
    yolo_device = models_dict.get('yolo_device', get_ultralytics_device(device))
    mc_models = models_dict['mc_models']
    mc_transform = models_dict['mc_transform']
    siamese = models_dict['siamese']
    
    # Reset tracker to prevent state leakage across videos
    print(f"  Resetting YOLO tracker for new video...")
    try:
        # Method 1: reset via predictor.tracker (newer ultralytics)
        if hasattr(yolo, "predictor") and yolo.predictor is not None:
            if hasattr(yolo.predictor, "tracker") and yolo.predictor.tracker is not None:
                yolo.predictor.tracker.reset()
                print(f"  ✓ Tracker reset successful (predictor.tracker)")
            elif hasattr(yolo.predictor, "trackers") and yolo.predictor.trackers:
                # Some versions use trackers (plural)
                for tracker in yolo.predictor.trackers.values():
                    if hasattr(tracker, 'reset'):
                        tracker.reset()
                print(f"  ✓ Tracker reset successful (predictor.trackers)")
            else:
                print(f"  (tracker not exposed in this ultralytics version; will reset via predictor=None)")
        # Method 2: reset via model.tracker (older ultralytics)
        elif hasattr(yolo, "tracker") and yolo.tracker is not None:
            yolo.tracker.reset()
            print(f"  ✓ Tracker reset successful (model.tracker)")
        else:
            print(f"  ⚠️ No tracker found to reset")
            
        # Method 3: force re-initialisation
        yolo.predictor = None  # force re-init on next call
        print(f"  ✓ Forced predictor reset")
        
    except Exception as e:
        print(f"  ⚠️ Could not reset YOLO tracker: {e}")
        print(f"  Proceeding anyway, but tracker state may be contaminated...")

    # Video I/O
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    #first_frame_of_video = True
    # Max frames to process
    max_frames = Config.MAX_VIDEO_FRAMES
    print(f"  Video settings: {fps:.1f} fps, processing up to {max_frames} frames")

    # Adaptive thresholds: scale proportionally to actual vs. reference resolution
    ref_w, ref_h = Config.REF_RESOLUTION
    res_scale = (w * h) / (ref_w * ref_h)
    adaptive_min_cell_area = max(100, int(Config.MIN_CELL_AREA * res_scale))
    # Linear scale for visual elements (box thickness, font size)
    vis_scale = res_scale ** 0.5   # sqrt: area ratio -> linear ratio
    adaptive_box_thickness  = max(1, round(5   * vis_scale))
    adaptive_font_scale     = max(0.3, round(1.5 * vis_scale * 10) / 10)
    adaptive_font_thickness = max(1, round(3   * vis_scale))
    print(f"  Resolution: {w}×{h} | res_scale={res_scale:.4f} | vis_scale={vis_scale:.3f}"
          f" | MIN_CELL_AREA={adaptive_min_cell_area}"
          f" | box_thickness={adaptive_box_thickness}"
          f" | font_scale={adaptive_font_scale}"
          f" | font_thickness={adaptive_font_thickness}")
    
    invalid_ids = set()
    used_ids = set()
    next_id = 10000
    initial_labels = {}
    initial_images = {}
    initial_tensors = {}  # cached initial tensors
    raw_sickle_predictions = {}
    all_frame_data = []

    # IDs from the first frame
    first_frame_ids = set()
    is_first_frame = True

    # Last known center per track {tid: (cx, cy, frame_id)}
    last_known_pos = {}
    # ID remap: restore canonical ID when YOLO assigns a new one
    id_remap = {}
    # Spatial jump threshold (px): skip if center moves > this in consecutive frames
    MAX_JUMP_PX = 200
    # ID recovery search radius (px)
    MAX_RECOVERY_PX = 200
    # Only attempt recovery within RECOVERY_BUFFER frames after loss
    RECOVERY_BUFFER = int(60)

    # Create transform once
    siamese_eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frame_id = 0

    print(f"  First pass: collecting raw predictions...")
    
    # Pass 1: collect all raw predictions
    while cap.isOpened():
        if frame_id >= max_frames:
            print(f"    Reached max_frames limit: {max_frames}")
            break
        ret, frame = cap.read()
        if not ret:
            print(f"    Video ended at frame {frame_id} ({frame_id/fps:.1f}s)")
            break

        orig = frame.copy()
        try:
            if frame_id == 0:
                print("\n  Tracker info (frame 0):")
                if hasattr(yolo, 'predictor') and yolo.predictor is not None:
                    if hasattr(yolo.predictor, 'trackers') and len(yolo.predictor.trackers) > 0:
                        tracker = yolo.predictor.trackers[0]
                        print(f"    - tracker type: {type(tracker).__name__}")
                        print(f"    - tracker module: {type(tracker).__module__}")
                        if hasattr(tracker, 'args'):
                            print(f"    - tracker args: {tracker.args}")
                print()

            _tracker_yaml = os.path.join(CELLBOX_MODELS_DIR, 'configs', 'botsort_cell.yaml')
            res = yolo.track(source=orig, persist=True, conf=Config.DETECTION_CONFIDENCE, iou=Config.YOLO_IOU,
                             max_det=2000, tracker=_tracker_yaml, device=yolo_device)[0]

            # persist_flag = (not first_frame_of_video)
            # res = yolo.track(source=orig, persist=persist_flag, conf=Config.DETECTION_CONFIDENCE, iou=0.3)[0]
            # first_frame_of_video = False

            boxes, ids = filter_boxes(res, w, h, min_cell_area=adaptive_min_cell_area)
        except Exception as e:
            print(f"    Warning: YOLO tracking failed at frame {frame_id}: {e}")
            frame_id += 1
            continue
            
        frame_detections = []
        skipped_boxes = []  # boxes skipped this frame (drawn as black in vis)
        # Collect cells for Siamese inference this frame
        siamese_batch_tids   = []
        siamese_batch_init   = []   # initial_tensor list
        siamese_batch_final  = []   # final_tensor list
        siamese_batch_ctype  = []   # cell_type index list

        for (x1, y1, x2, y2), orig_id in zip(boxes, ids):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # --- Resolve raw YOLO ID; handle invalid_ids first ---
            if orig_id in invalid_ids:
                while next_id in used_ids: next_id += 1
                tid = next_id
            else:
                # Check for a known ID remap
                tid = id_remap.get(orig_id, orig_id)
            used_ids.add(tid)

            if is_first_frame:
                first_frame_ids.add(tid)
                last_known_pos[tid] = (cx, cy, frame_id)
            else:
                if tid not in first_frame_ids:
                    # --- ID recovery: find nearest lost track by distance ---
                    recovered_tid = None
                    best_dist = MAX_RECOVERY_PX
                    for known_tid, (lx, ly, lf) in last_known_pos.items():
                        if known_tid not in first_frame_ids:
                            continue
                        if frame_id - lf > RECOVERY_BUFFER:
                            continue
                        dist = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            recovered_tid = known_tid
                    if recovered_tid is not None:
                        id_remap[orig_id] = recovered_tid
                        tid = recovered_tid
                        print(f"    [ID recovery] new {orig_id} -> {recovered_tid} (dist={best_dist:.1f}px, frame={frame_id})")
                    else:
                        skipped_boxes.append((x1, y1, x2, y2))
                        continue

                # --- Spatial jump check ---
                if tid in last_known_pos:
                    lx, ly, lf = last_known_pos[tid]
                    if lf == frame_id - 1:  # strictly consecutive frames only
                        jump = ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5
                        if jump > MAX_JUMP_PX:
                            print(f"    [spatial jump] ID {tid} at frame {frame_id} moved {jump:.1f}px > {MAX_JUMP_PX}px, skipped")
                            skipped_boxes.append((x1, y1, x2, y2))
                            continue

            crop = orig[y1:y2, x1:x2]
            if crop.size == 0:
                invalid_ids.add(orig_id)
                skipped_boxes.append((x1, y1, x2, y2))
                continue

            pil_rgb = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            gray = pil_rgb.convert("L")
            proc = pil_rgb

            if tid not in initial_labels:
                # Step 1: 7-class EfficientNet classification
                inp = mc_transform(proc).unsqueeze(0).to(device)
                outm = sum(m(inp) for m in mc_models) / len(mc_models)
                base_lbl = MC_CLASSES[torch.argmax(torch.softmax(outm,1),1).item()]

                # Step 2: aspect-ratio reclassification for Dis/ISC
                if Config.USE_SEG_RECLASSIFY and seg_model is not None and base_lbl in ['Dis', 'ISC']:
                    reclassified_lbl, aspect_ratio = reclassify_by_aspect_ratio(
                        crop, seg_model, base_lbl, conf_threshold=0.05,
                        aspect_ratio_threshold=Config.SEG_ASPECT_RATIO_THRESHOLD,
                        yolo_device=yolo_device,
                    )
                    if reclassified_lbl != base_lbl and aspect_ratio is not None:
                        print(f"    Track {tid}: Reclassified {base_lbl} → {reclassified_lbl} (AR={aspect_ratio:.2f})")
                    base_lbl = reclassified_lbl

                # Step 3: filter edge-truncated Ret detections in first frame
                if is_first_frame and base_lbl == 'Ret' and base_lbl == "ISC":
                    at_edge = (x1 < Config.MARGIN or y1 < Config.MARGIN or
                               x2 > w - Config.MARGIN or y2 > h - Config.MARGIN)
                    if at_edge:
                        first_frame_ids.discard(tid)
                        print(f"    [edge Ret filter] Track {tid} at edge, skipped")
                        skipped_boxes.append((x1, y1, x2, y2))
                        continue

                initial_labels[tid] = base_lbl
                initial_images[tid] = (proc, gray)
                initial_tensors[tid] = siamese_eval_transform(proc).unsqueeze(0).to(device)
                raw_sickle_predictions[tid] = [False] * max_frames
            else:
                base_lbl = initial_labels[tid]
                # Append to batch; defer inference
                siamese_batch_tids.append(tid)
                siamese_batch_init.append(initial_tensors[tid])
                siamese_batch_final.append(siamese_eval_transform(proc).unsqueeze(0))
                siamese_batch_ctype.append(six2siamese[base_lbl])

            # Update last known position
            last_known_pos[tid] = (cx, cy, frame_id)

            frame_detections.append({
                'track_id': tid,
                'box': (x1, y1, x2, y2),
                'base_label': base_lbl,
                'crop': crop,
                'proc': proc,
                'gray': gray
            })

        # Batch Siamese inference for all cells in this frame
        if siamese_batch_tids:
            batch_init  = torch.cat(siamese_batch_init, dim=0).to(device)
            batch_final = torch.cat(siamese_batch_final, dim=0).to(device)
            batch_ctype = torch.tensor(siamese_batch_ctype, dtype=torch.long).to(device)
            with torch.no_grad():
                batch_logits = siamese(batch_init, batch_final, batch_ctype)
                batch_probs  = torch.sigmoid(batch_logits).cpu().numpy().flatten()
            for tid, prob in zip(siamese_batch_tids, batch_probs):
                raw_sickle_predictions[tid][frame_id] = (prob >= Config.CLASSIFICATION_CONFIDENCE)
                if tid == 174:
                    lbl = initial_labels.get(174, '?')
                    is_s = prob >= Config.CLASSIFICATION_CONFIDENCE
                    print(f"  [DEBUG id=174] frame={frame_id:3d}  label={lbl}  prob={prob:.4f}  sickle={is_s}")

        all_frame_data.append({
            'frame_id': frame_id,
            'orig_frame': orig,
            'detections': frame_detections,
            'skipped_boxes': skipped_boxes
        })
        
        if is_first_frame:
            is_first_frame = False
            print(f"    First frame done: {len(first_frame_ids)} valid IDs: {sorted(first_frame_ids)}")
            print(f"  ===== First frame cells: {len(first_frame_ids)} | IOU={Config.YOLO_IOU} | conf={Config.DETECTION_CONFIDENCE} =====")
        
        frame_id += 1
        if frame_id % 500 == 0:
            print(f"    Processed {frame_id} frames ({frame_id/fps:.1f}s)...")
            
    print(f"    Video processing completed: {frame_id} frames ({frame_id/fps:.1f}s total)")

    cap.release()

    if Config.USE_RET_CONSTRAINT:
        print(f"  Applying Ret-specific constraints...")
        constrained_preds = apply_ret_constraint(
            raw_sickle_predictions,
            all_frame_data,
            initial_images,
            initial_labels,
            device,
            siamese,
            fps
        )
    else:
        print(f"  Skipping Ret-specific constraints (USE_RET_CONSTRAINT=False)")
        constrained_preds = {tid: preds.copy() for tid, preds in raw_sickle_predictions.items()}

    time_windows_frames = Config.get_time_windows_frames(fps)
    print(f"  Time windows (frames @ {fps:.1f}fps): { {c: time_windows_frames[c] for c in MC_CLASSES} }")
    print(f"  Applying time window smoothing...")
    smoothed_sickle_predictions = apply_time_window_smoothing(
        constrained_preds,
        initial_labels=initial_labels,
        time_windows_frames=time_windows_frames
    )

    print(f"  Applying Ech and Ret similarity constraints (post-smoothing)...")
    final_sickle_predictions = apply_ech_ret_similarity_constraint(
        smoothed_sickle_predictions,
        all_frame_data,
        initial_images,
        initial_labels,
        device,
        siamese,
        fps
    )

    prediction_stages = {
        'raw': raw_sickle_predictions,
        'ret': constrained_preds,
        'smoothed': smoothed_sickle_predictions,
        'final': final_sickle_predictions,
    }
    vis_source = str(Config.VISUALIZATION_SICKLE_SOURCE).strip().lower()
    if vis_source not in prediction_stages:
        print(f"  Warning: invalid VISUALIZATION_SICKLE_SOURCE='{Config.VISUALIZATION_SICKLE_SOURCE}', fallback to 'final'")
        vis_source = 'final'
    vis_sickle_predictions = prediction_stages[vis_source]
    print(f"  Visualization sickle source: {vis_source}")

    print(f"  Generating statistics and output video...")



    # Generate output video
    video_name = f"video_{os.path.splitext(os.path.basename(video_path))[0]}"
    out = cv2.VideoWriter(os.path.join(output_dir, f"{video_name}.mp4"),
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (w,h))
    
    records = []
    frame_stats = []
    
    # Fixed cell totals from the first frame
    first_frame_totals = {}
    first_frame_calculated = False
    
    # Track per-cell sickle status to enforce monotonicity
    cell_sickle_status = {}  # {track_id: current_sickle_status}
    all_track_ids = set()
    
    # Pre-scan: collect all track IDs and classes
    track_id_to_class = {}
    for frame_data in all_frame_data:
        for detection in frame_data['detections']:
            tid = detection['track_id']
            base_lbl = detection['base_label']
            
            if tid not in first_frame_ids:
                continue
                
            all_track_ids.add(tid)
            track_id_to_class[tid] = base_lbl
            if tid not in cell_sickle_status:
                cell_sickle_status[tid] = False
    
    print(f"    Total unique track IDs: {len(all_track_ids)}")
    class_counts = {}
    for tid, cls in track_id_to_class.items():
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"    Track ID distribution: {class_counts}")
    
    for frame_data in all_frame_data:
        frame_id = frame_data['frame_id']
        orig = frame_data['orig_frame']
        detections = frame_data['detections']
        
        vis = orig.copy()
        
        # Count sickle cells correctly (detected cells only)
        frame_stat = {'frame': frame_id, 'time_s': frame_id / fps}
        
        current_sickle_counts = {}
        for c in MC_CLASSES:
            current_sickle_counts[c] = 0
        
        current_frame_tracks = set()
        
        for detection in detections:
            tid = detection['track_id']
            x1, y1, x2, y2 = detection['box']
            base_lbl = detection['base_label']
            
            if tid not in first_frame_ids:
                continue
                
            current_frame_tracks.add(tid)
            
            vis_track_preds = vis_sickle_predictions.get(tid, [])
            predicted_sickle = vis_track_preds[frame_id] if frame_id < len(vis_track_preds) else False

            raw_track_preds = raw_sickle_predictions.get(tid, [])
            ret_track_preds = constrained_preds.get(tid, [])
            smoothed_track_preds = smoothed_sickle_predictions.get(tid, [])
            final_track_preds = final_sickle_predictions.get(tid, [])

            raw_is_sickle = raw_track_preds[frame_id] if frame_id < len(raw_track_preds) else False
            ret_is_sickle = ret_track_preds[frame_id] if frame_id < len(ret_track_preds) else False
            smoothed_is_sickle = smoothed_track_preds[frame_id] if frame_id < len(smoothed_track_preds) else False
            final_is_sickle = final_track_preds[frame_id] if frame_id < len(final_track_preds) else False
            
            # Enforce sickle monotonicity
            if predicted_sickle or cell_sickle_status[tid]:
                is_sickle = True
                cell_sickle_status[tid] = True
            else:
                is_sickle = False
            
            if is_sickle:
                current_sickle_counts[base_lbl] += 1
            
            label = f"sickle_{base_lbl}" if is_sickle else base_lbl
            color = DEO_COLOR if is_sickle else COLORS[base_lbl]

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, adaptive_box_thickness)
            cv2.putText(vis, f"ID{tid}:{label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, adaptive_font_scale, color, adaptive_font_thickness)
            
            records.append({
                'video_idx': 1,
                'frame': frame_id,
                'track_id': tid,
                'base_label': base_lbl,
                'is_sickle': is_sickle,
                'raw_is_sickle': raw_is_sickle,
                'ret_is_sickle': ret_is_sickle,
                'smoothed_is_sickle': smoothed_is_sickle,
                'final_is_sickle': final_is_sickle,
                'visualization_source': vis_source,
            })

        # Draw skipped boxes in black
        for sx1, sy1, sx2, sy2 in frame_data.get('skipped_boxes', []):
            cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), (0, 0, 0), adaptive_box_thickness)

        # (do not count undetected cells as sickle)
        # for tid in all_track_ids:
        #     if tid not in current_frame_tracks and cell_sickle_status[tid]:
        #         base_lbl = track_id_to_class[tid]
        #         current_sickle_counts[base_lbl] += 1

        try:
            for c in MC_CLASSES:
                frame_stat[f"{c}_total"] = first_frame_totals.get(c, 0) if first_frame_calculated else 0
                frame_stat[f"{c}_sickle"] = current_sickle_counts.get(c, 0)
        except Exception as e:
            print(f"    Error setting frame_stat at frame {frame_id}: {e}")
            print(f"    first_frame_calculated: {first_frame_calculated}")
            print(f"    first_frame_totals: {first_frame_totals}")
            print(f"    current_sickle_counts: {current_sickle_counts}")
            for c in MC_CLASSES:
                frame_stat[f"{c}_total"] = 0
                frame_stat[f"{c}_sickle"] = 0

        if not first_frame_calculated:
            try:
                initial_cells_by_class = {c: set() for c in MC_CLASSES}  # {class: {track_id, ...}}
                
                for frame_data in all_frame_data:
                    if frame_data['frame_id'] >= Config.INITIAL_FRAMES_COUNT:
                        break  # only first N frames
                    
                    for detection in frame_data['detections']:
                        tid = detection['track_id']
                        base_lbl = detection['base_label']
                        if base_lbl in initial_cells_by_class:
                            initial_cells_by_class[base_lbl].add(tid)
                
                first_frame_totals = {c: len(initial_cells_by_class[c]) for c in MC_CLASSES}
                first_frame_calculated = True
                print(f"    Collected initial cell totals from first {Config.INITIAL_FRAMES_COUNT} frames: {first_frame_totals}")
                
                for c in MC_CLASSES:
                    frame_stat[f"{c}_total"] = first_frame_totals.get(c, 0)
                    
            except Exception as e:
                print(f"    Error calculating initial frame totals: {e}")
                first_frame_totals = {c: 0 for c in MC_CLASSES}
                first_frame_calculated = True

        #if frame_id % 900 == 0:
        #if int(fps) > 0 and frame_id % int(30 * fps) == 0:
        if int(fps) > 0 and frame_id > 0 and frame_id % int(30 * fps) == 0:

            time_s = frame_id / fps
            sickle_counts = {c: frame_stat[f"{c}_sickle"] for c in MC_CLASSES if frame_stat[f"{c}_total"] > 0}
            total_counts = {c: frame_stat[f"{c}_total"] for c in MC_CLASSES if frame_stat[f"{c}_total"] > 0}
            rates = {c: f"{frame_stat[f'{c}_sickle']/first_frame_totals.get(c, 1):.3f}" for c in MC_CLASSES if first_frame_totals.get(c, 0) > 0}
            print(f"    Frame {frame_id} ({time_s:.0f}s): Sickle={sickle_counts}, Rates={rates}")

        out.write(vis)
        frame_stats.append(frame_stat)

    out.release()
    
    frame_stats = verify_and_fix_monotonicity(frame_stats, MC_CLASSES)
    
    print(f"  Final sickle rate analysis for video:")
    
    if not frame_stats:
        print(f"    Warning: No frame stats generated for video")
        return [], {}
    
    if not first_frame_totals:
        print(f"    Warning: No first frame totals set for video")
        return frame_stats, {}
    
    try:
        for c in MC_CLASSES:
            if first_frame_totals.get(c, 0) > 0:
                final_sickle = frame_stats[-1][f"{c}_sickle"] if frame_stats else 0
                total = first_frame_totals[c]
                final_rate = final_sickle / total
                print(f"    {c}: {final_sickle}/{total} = {final_rate:.3f} ({final_rate*100:.1f}%)")
    except Exception as e:
        print(f"    Error calculating final rates: {e}")
        print(f"    Frame stats length: {len(frame_stats)}")
        print(f"    First frame totals: {first_frame_totals}")
        if frame_stats:
            print(f"    Last frame keys: {list(frame_stats[-1].keys()) if frame_stats else 'None'}")
    
    # Save records CSV
    df_video = pd.DataFrame(records)
    df_video.to_csv(os.path.join(output_dir, f"{video_name}_records.csv"), index=False)
    
    label_counts = Counter(initial_labels.values())
    
    rates = []
    for frame_stat in frame_stats:
        rate_stat = {'frame': frame_stat['frame'], 'time_s': frame_stat['time_s']}
        for c in MC_CLASSES:
            total = first_frame_totals.get(c, 0)
            sickle = frame_stat[f"{c}_sickle"]
            rate_stat[f"{c}_rate"] = sickle / total if total > 0 else 0.0
        rates.append(rate_stat)
    
    df_rates = pd.DataFrame(rates)
    
    print(f"  Video complete. Generated: {video_name}.mp4, {video_name}_records.csv")
    print(f"  Unique cells by class: {dict(label_counts)}")
    print(f"  Total frames processed: {len(frame_stats)}")
    
    return frame_stats, label_counts

def records_to_stats_cumulative(records_csv: str, fps: float = 4.0) -> pd.DataFrame:
    """Build cumulative sickle stats from *_records.csv.
    Columns: frame, time_s, <c>_sickle, <c>_total (fixed to frame-0 count).
    Once a track becomes sickle, its count never decreases.
    """
    print(f"  Processing: {os.path.basename(records_csv)}")
    df = pd.read_csv(records_csv)
    
    df['is_sickle'] = (df['is_sickle'].astype(str).str.lower()
                       .map({'true': 1, 'false': 0}).fillna(0).astype(int))
    
    print(f"    Original data: {len(df)} records, {df['track_id'].nunique()} unique tracks")
    
    # Find first sickle frame per track
    sickle_start_frames = {}
    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id].sort_values('frame')
        base_label = track_data['base_label'].iloc[0]
        sickle_frames = track_data[track_data['is_sickle'] == 1]['frame']
        if len(sickle_frames) > 0:
            sickle_start_frames[track_id] = {
                'start_frame': sickle_frames.min(),
                'base_label': base_label
            }
    
    print(f"    Found {len(sickle_start_frames)} tracks that become sickle")
    
    # Per-frame cumulative sickle count
    frames = sorted(df['frame'].unique())
    stats_data = []
    
    for frame in frames:
        frame_stats = {'frame': frame, 'time_s': frame / float(fps)}
        
        for c in MC_CLASSES:
            cumulative_sickle = 0
            for track_id, info in sickle_start_frames.items():
                if info['base_label'] == c and frame >= info['start_frame']:
                    cumulative_sickle += 1
            frame_stats[f'{c}_sickle'] = cumulative_sickle
        
        stats_data.append(frame_stats)
    
    # Get frame-0 totals
    totals_frame = (df.groupby(['frame','base_label'])['track_id']
                      .nunique().unstack('base_label', fill_value=0))
    first_totals = totals_frame.iloc[0] if len(totals_frame) else pd.Series(0, index=MC_CLASSES)
    
    for frame_stat in stats_data:
        for c in MC_CLASSES:
            frame_stat[f'{c}_total'] = int(first_totals.get(c, 0))
    
    stats = pd.DataFrame(stats_data)
    
    need = ['frame','time_s'] + [f'{c}_sickle' for c in MC_CLASSES] + [f'{c}_total' for c in MC_CLASSES]
    for col in need:
        if col not in stats: stats[col] = 0
    
    print(f"    Generated cumulative stats table: {len(stats)} frames")
    return stats[need].sort_values('frame')

def pool_cumulative_from_csvs(records_csv_list: list, fps: float = 4.0, duration_s: float = 120.0) -> pd.DataFrame:
    """Pool first-sickle frames from all videos; compute one combined cumulative curve.
    Equivalent to summing per-video cumulative curves, but more direct.
    """
    all_events_by_class = {c: [] for c in MC_CLASSES}  # {class: [first_sickle_frame, ...]}
    totals = {c: 0 for c in MC_CLASSES}
    max_event_frame = 0
    max_data_frame = 0  # actual last frame in data

    for csv_path in records_csv_list:
        print(f"  Pooling: {os.path.basename(csv_path)}")
        df = pd.read_csv(csv_path)
        df['is_sickle'] = (df['is_sickle'].astype(str).str.lower()
                           .map({'true': 1, 'false': 0}).fillna(0).astype(int))

        max_data_frame = max(max_data_frame, int(df['frame'].max()))

        # Accumulate first-frame totals per class
        first_frame = int(df['frame'].min())
        first_df = df[df['frame'] == first_frame]
        for c in MC_CLASSES:
            totals[c] += int(first_df[first_df['base_label'] == c]['track_id'].nunique())

        for track_id, grp in df.groupby('track_id'):
            base_label = grp.sort_values('frame')['base_label'].iloc[0]
            if base_label not in MC_CLASSES:
                continue
            sickle_frames = grp[grp['is_sickle'] == 1]['frame']
            if len(sickle_frames) > 0:
                first_sf = int(sickle_frames.min())
                all_events_by_class[base_label].append(first_sf)
                max_event_frame = max(max_event_frame, first_sf)

    print(f"  Pooled totals: { {c: totals[c] for c in MC_CLASSES} }")
    for c in MC_CLASSES:
        print(f"    {c}: {len(all_events_by_class[c])} sickle events / {totals[c]} total")

    # Use actual data range, not duration_s * fps
    max_frame = max(max_data_frame, max_event_frame)
    frame_grid = np.arange(0, max_frame + 1, dtype=int)
    time_grid = frame_grid / float(fps)

    # Compute per-frame cumulative counts via searchsorted
    sickle_arrays = {}
    for c in MC_CLASSES:
        if all_events_by_class[c]:
            sorted_events = np.sort(np.array(all_events_by_class[c]))
            sickle_arrays[c] = np.searchsorted(sorted_events, frame_grid, side='right').astype(float)
        else:
            sickle_arrays[c] = np.zeros(len(frame_grid), dtype=float)

    out = pd.DataFrame({'frame': frame_grid, 'time_s': time_grid})
    for c in MC_CLASSES:
        out[f'{c}_sickle'] = sickle_arrays[c]
        out[f'{c}_total'] = totals[c]
        out[f'{c}_rate'] = sickle_arrays[c] / totals[c] if totals[c] > 0 else 0.0

    return out


def combine_by_frame_cumulative(per_video_stats, fps: float = 4.0, duration_s: float = 120.0) -> pd.DataFrame:
    """Align multiple videos by frame index (same fps). Hold last value after video ends."""
    # Unified frame grid
    max_frames = int(duration_s * fps)
    frame_grid = np.arange(0, max_frames + 1)
    time_grid  = frame_grid / float(fps)

    # Fixed totals = sum of first-frame totals across all videos
    totals = {c: 0 for c in MC_CLASSES}
    for i, df in enumerate(per_video_stats):
        print(f"    Video {i+1} first-frame counts:")
        for c in MC_CLASSES:
            first_frame_count = int(df.iloc[0][f'{c}_total'])
            totals[c] += first_frame_count
            print(f"      {c}: {first_frame_count}")
    
    print("    Combined fixed totals:")
    for c in MC_CLASSES:
        print(f"      {c}: {totals[c]}")

    # Sum sickle counts; hold last value after video ends
    summed = {c: np.zeros_like(frame_grid, dtype=float) for c in MC_CLASSES}
    for df in per_video_stats:
        video_end_frame = df['frame'].max()
        
        dfi = df.set_index('frame').reindex(frame_grid)
        for c in MC_CLASSES:
            col = f'{c}_sickle'
            if col in dfi:
                last_value = df[df['frame'] == video_end_frame][col].iloc[0] if video_end_frame in df['frame'].values else 0
                dfi[col] = dfi[col].fillna(0.0)
                for frame_idx in range(len(frame_grid)):
                    frame = frame_grid[frame_idx]
                    if frame > video_end_frame:
                        dfi.iloc[frame_idx, dfi.columns.get_loc(col)] = last_value
                summed[c] += dfi[col].to_numpy()
            else:
                summed[c] += np.zeros_like(frame_grid, dtype=float)

    out = pd.DataFrame({'frame': frame_grid, 'time_s': time_grid})
    for c in MC_CLASSES:
        out[f'{c}_sickle'] = summed[c]
        out[f'{c}_total']  = totals[c]
        out[f'{c}_rate']   = (summed[c] / totals[c]) if totals[c] else 0.0
    return out

def combine_multi_video_csv_data(records_csv_list, fps: float = 4.0, duration_s: float = 120.0, output_root=None):
    """Merge per-video CSV data and generate combined analysis plots.

    Args:
        records_csv_list: Absolute paths to per-video records.csv files
    """
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    _default_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    out_dir = os.path.join(output_root or _default_root, 'combined_analysis_output')
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Output directory: {out_dir}")

    records_paths = []
    for p in records_csv_list:
        if os.path.exists(p):
            records_paths.append(p)
            print(f"  Using records file: {os.path.basename(p)}")
        else:
            print(f"  WARNING: file not found: {p}")

    if len(records_paths) < 2:
        print(f"  WARNING: not enough valid records files ({len(records_paths)}), skipping")
        return
    
    print(f"\n  === Processing {len(records_paths)} videos (cumulative sickle) ===")
    valid_paths = []
    per_video_stats = []
    for i, p in enumerate(records_paths, 1):
        if os.path.exists(p):
            stats = records_to_stats_cumulative(p, fps=fps)
            per_video_stats.append(stats)
            valid_paths.append(p)
        else:
            print(f"    WARNING: Video {i} file not found: {p}")

    if per_video_stats:
        print("\n  === Cross-video pooled merge ===")
        df_combined = pool_cumulative_from_csvs(valid_paths, fps=fps, duration_s=duration_s)
        combined_csv = os.path.join(out_dir, f'combined_sickle_rates_cumulative_{timestamp}.csv')
        df_combined.to_csv(combined_csv, index=False)
        print(f'    ✓ Saved: {combined_csv}')

        def _plot_cumulative(df_plot, png_path):
            try:
                legend_items = []
                for c in MC_CLASSES:
                    total_col = f'{c}_total'
                    count = int(df_plot[total_col].iloc[0]) if total_col in df_plot.columns else 0
                    if count > 0:
                        letter = CLASS_LETTERS[c]
                        legend_items.append((letter, c, f"{letter} ({count} cells)", COLOR_MAP[c]))
                legend_items.sort()

                x_max = Config.MAX_VIDEO_FRAMES / fps
                _, ax = plt.subplots(figsize=(12, 8))
                for _, c, label, color in legend_items:
                    ax.plot(df_plot['time_s'], df_plot[f'{c}_rate'] * 100,
                            label=label, color=color, linewidth=2, drawstyle='steps-post')
                ax.set_xlim(0, x_max)
                ax.set_ylim(0, 100)
                ax.set_xticks(np.arange(0, x_max + 1, 20))
                ax.set_yticks(np.arange(0, 101, 20))
                ax.tick_params(axis='both', which='major', labelsize=22, width=1.5, length=6)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.set_xlabel('Time (s)', fontsize=24)
                ax.set_ylabel('Sickled fraction (%)', fontsize=24)
                ax.legend(loc='upper left', fontsize=22)
                plt.subplots_adjust(left=0.09, bottom=0.09, right=0.98, top=0.97)
                plt.savefig(png_path, dpi=150)
                plt.close()
                print(f'    ✓ Saved: {png_path}')
            except Exception as e:
                print(f'    ⚠️ Plot failed: {e}')

        print("\n  === Generating combined curve ===")
        _plot_cumulative(
            df_combined,
            png_path=os.path.join(out_dir, f'combined_sickle_rate_cumulative_{timestamp}.png')
        )

        # combined binary plot: total sickled fraction across all classes
        total_sickle = sum(df_combined[f'{c}_sickle'] for c in MC_CLASSES)
        total_cells  = sum(df_combined[f'{c}_total']  for c in MC_CLASSES)
        binary_rate  = np.where(total_cells > 0, total_sickle / total_cells * 100, 0.0)
        plt.figure(figsize=(8, 5))
        plt.plot(df_combined['time_s'], binary_rate,
                 label='Total sickled fraction', color='black')
        plt.xlabel('Time (s)')
        plt.ylabel('Sickled fraction (%)')
        plt.ylim(0, 100)
        plt.title('Total cell ratio (binary) — combined')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        binary_png = os.path.join(out_dir, 'state_ratio_plot_binary.png')
        plt.savefig(binary_png, dpi=300)
        plt.close()
        pd.DataFrame({'Time_sec': df_combined['time_s'],
                      'Sickled_fraction_percent (%)': binary_rate}
                     ).to_csv(binary_png.replace('.png', '.csv'), index=False)
        print(f'    ✓ state_ratio_plot_binary.png')
    else:
        print("    WARNING: No valid data files found!")


def main(video_path, model_paths, output_root=None):
    """Process a single video.

    Args:
        video_path: Path to input video
        model_paths: Dict with keys yolo, seg, mc_base_dir, mc_prefix, siamese
        output_root: Root output directory; auto-generated if None
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    name = f"single_video_{video_name}_{now}"
    device = get_torch_device()
    yolo_device = get_ultralytics_device(device)

    _default_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')
    output_dir = os.path.join(output_root or _default_root, f"results_{name}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    print("Loading models...")
    print(f"  YOLO:    {model_paths['yolo']}")
    print(f"  Siamese: {model_paths['siamese']}")

    yolo_path = model_paths['yolo']
    siamese_path = model_paths['siamese']
    
    try:
        # YOLO
        print("  Loading YOLO model...")
        yolo = YOLO(yolo_path)

        print("  Tracker info:")
        if hasattr(yolo, 'predictor') and yolo.predictor is not None:
            if hasattr(yolo.predictor, 'args'):
                tracker_config = getattr(yolo.predictor.args, 'tracker', 'Not specified')
                print(f"    - tracker config: {tracker_config}")
            if hasattr(yolo.predictor, 'trackers'):
                print(f"    - trackers: {yolo.predictor.trackers}")
                if len(yolo.predictor.trackers) > 0:
                    tracker_obj = yolo.predictor.trackers[0]
                    print(f"    - tracker type: {type(tracker_obj).__name__}")
        print(f"    - YOLO module: {yolo.__class__.__module__}")

        # Segmentation model for aspect-ratio reclassification
        if Config.USE_SEG_RECLASSIFY:
            print("  Loading segmentation model for aspect ratio reclassification...")
            seg_model_path = model_paths['seg']
            seg_model = YOLO(seg_model_path)
            print(f"    Loaded: {os.path.basename(seg_model_path)}")
        else:
            seg_model = None
            print("  Segmentation model skipped (USE_SEG_RECLASSIFY=False)")

        # Multi-class models (EfficientNet_b3 x5 folds)
        print("  Loading multi-class classification models...")
        mc_models, mc_transform = [], None
        mc_base_dir = model_paths['mc_base_dir']
        mc_prefix   = model_paths['mc_prefix']
        mc_n_folds  = model_paths.get('mc_n_folds', 5)
        for i in range(1, mc_n_folds + 1):
            p = os.path.join(mc_base_dir, f"{mc_prefix}{i}_best.pth")
            if os.path.exists(p):
                print(f"    Loading: {os.path.basename(p)}")
                m, mc_transform = get_model_and_transform('EfficientNet_b3', p, len(MC_CLASSES), device)
                mc_models.append(m)
            else:
                print(f"    Warning: Model file not found: {p}")

        if not mc_models:
            raise FileNotFoundError("No multi-class models loaded successfully!")
        print(f"    Total MC models loaded: {len(mc_models)}")

        print("  Loading Siamese model...")
        if os.path.exists(siamese_path):
            checkpoint = torch.load(siamese_path, map_location=device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                fusion_mode = checkpoint.get('config', {}).get('fusion_mode', 'B')
            else:
                state_dict = checkpoint
                fusion_mode = 'B'
            
            if 'classifier.0.weight' in state_dict:
                classifier_in_dim = state_dict['classifier.0.weight'].shape[1]
                if classifier_in_dim == 1024:  # 512*2
                    fusion_mode = 'A' if 'film_layer4.type_embed.weight' not in state_dict else 'C'
                elif classifier_in_dim == 1025:  # 512*2+1
                    fusion_mode = 'B'
                print(f"    Detected fusion mode: {fusion_mode} (classifier input: {classifier_in_dim})")
            
            use_film = 'feature_extractor.film_layer4.type_embed.weight' in state_dict
            use_cbam = 'feature_extractor.cbam_layer1.ca.fc1.weight' in state_dict
            print(f"    Model config: FiLM={use_film}, CBAM={use_cbam}, Fusion={fusion_mode}")
            
            siamese = ConditionalSiameseNetwork(
                num_cell_types=7, 
                use_film=use_film, 
                use_cbam=use_cbam, 
                fusion_mode=fusion_mode
            )
            
            siamese.load_state_dict(state_dict)
            print(f"    Loaded: {os.path.basename(siamese_path)}")
        else:
            raise FileNotFoundError(f"Siamese model not found: {siamese_path}")
        siamese.to(device).eval()

        models_dict = {
            'yolo': yolo,
            'seg_model': seg_model,
            'yolo_device': yolo_device,
            'mc_models': mc_models,
            'mc_transform': mc_transform,
            'siamese': siamese
        }
        
        print(f"  Successfully loaded:")
        print(f"    - YOLO model")
        print(f"    - Segmentation model for aspect ratio")
        print(f"    - {len(mc_models)} multi-class models")
        print(f"    - Siamese model")
        print(f"  Using PyTorch device: {device}")
        print(f"  Using Ultralytics device: {yolo_device}")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please check the model paths and ensure all files exist.")
        return

    print(f"Checking video file:")
    if not os.path.exists(video_path):
        print(f"  ✗ Video: {os.path.basename(video_path)} - Not found")
        print("Video file not found! Please check the video path.")
        return
    
    print(f"  ✓ Video: {os.path.basename(video_path)} - Found")

    if Config.AUTO_DET_CONF:
        print("Auto-detecting detection confidence from video...")
        Config.DETECTION_CONFIDENCE = auto_detect_conf(
            video_path,
            models_dict['yolo'],
            yolo_device=models_dict.get('yolo_device'),
        )
    print(f"  DETECTION_CONFIDENCE = {Config.DETECTION_CONFIDENCE}")

    # Per-video --max-time conversion (each video may have different fps)
    if getattr(Config, 'MAX_TIME_S', None) is not None:
        _cap_tmp = cv2.VideoCapture(video_path)
        _video_fps = _cap_tmp.get(cv2.CAP_PROP_FPS) or 4.0
        _cap_tmp.release()
        Config.MAX_VIDEO_FRAMES = max(1, int(Config.MAX_TIME_S * _video_fps))
        print(f"  --max-time {Config.MAX_TIME_S}s × {_video_fps:.1f}fps → MAX_VIDEO_FRAMES={Config.MAX_VIDEO_FRAMES}")

    print(f"\nExpected processing time: ~15-20 minutes (estimated)")

    frame_stats, label_counts = process_single_video(video_path, models_dict, device, output_dir)

    processed_video_stem = f"video_{os.path.splitext(os.path.basename(video_path))[0]}"

    print("Generating plots...")
    if frame_stats:
        plot_state_ratio_binary(frame_stats, output_dir)
    plot_frame0_class_pie(label_counts, output_dir)

    print("Generating sickle rate analysis...")

    legend_items = []
    for c in MC_CLASSES:
        count = label_counts.get(c, 0)
        if count > 0:
            letter = CLASS_LETTERS[c]
            label = f"{letter} ({count})"
            color = COLOR_MAP[c]
            legend_items.append((letter, c, label, color))
    legend_items.sort()

    print(f"\n{'='*60}")
    print("PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Processed video: {os.path.basename(video_path)}")
    print(f"Output directory: {output_dir}")
    
    print(f"\nCell counts in video:")
    total_unique_cells = sum(label_counts.values())
    for c in MC_CLASSES:
        count = label_counts.get(c, 0)
        print(f"  {display_name(c)}: {count} unique cells")
    print(f"  Total unique cells: {total_unique_cells}")
    
    if frame_stats:
        final_frame = frame_stats[-1]
        print(f"\nFinal sickle rates (at end of video):")
        for c in MC_CLASSES:
            if label_counts.get(c, 0) > 0:
                total = final_frame[f"{c}_total"]
                sickle = final_frame[f"{c}_sickle"] 
                rate = sickle / total if total > 0 else 0.0
                print(f"  {display_name(c)}: {sickle}/{total} = {rate:.3f} ({rate*100:.1f}%)")
    
    records_csv_path = os.path.join(output_dir, f"{processed_video_stem}_records.csv")
    try:
        fps_val = frame_stats[1]['time_s'] - frame_stats[0]['time_s'] if len(frame_stats) > 1 else 0.25
        fps_val = 1.0 / fps_val if fps_val > 0 else 4.0
        stats_cumul = records_to_stats_cumulative(records_csv_path, fps=fps_val)
        for c in MC_CLASSES:
            total = int(stats_cumul[f'{c}_total'].iloc[0]) if f'{c}_total' in stats_cumul.columns else 0
            stats_cumul[f'{c}_rate'] = stats_cumul[f'{c}_sickle'] / total if total > 0 else 0.0

        cumul_legend = []
        for c in MC_CLASSES:
            total = int(stats_cumul[f'{c}_total'].iloc[0]) if f'{c}_total' in stats_cumul.columns else 0
            if total > 0:
                cumul_legend.append((CLASS_LETTERS[c], c, f"{CLASS_LETTERS[c]} ({total})", COLOR_MAP[c]))
        cumul_legend.sort()

        x_max = Config.MAX_VIDEO_FRAMES / fps_val
        _, ax_c = plt.subplots(figsize=(12, 8))
        for _, c, label_txt, color in cumul_legend:
            total = int(stats_cumul[f'{c}_total'].iloc[0]) if f'{c}_total' in stats_cumul.columns else 0
            ax_c.plot(stats_cumul['time_s'], stats_cumul[f'{c}_rate'] * 100,
                      label=f"{CLASS_LETTERS[c]} ({total} cells)",
                      color=color, linewidth=2, drawstyle='steps-post')
        ax_c.set_xlim(0, x_max)
        ax_c.set_ylim(0, 100)
        ax_c.set_xticks(np.arange(0, x_max + 1, 20))
        ax_c.set_yticks(np.arange(0, 101, 20))
        ax_c.tick_params(axis='both', which='major', labelsize=22, width=1.5, length=6)
        ax_c.spines['right'].set_visible(False)
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['left'].set_linewidth(1.5)
        ax_c.spines['bottom'].set_linewidth(1.5)
        ax_c.set_xlabel('Time (s)', fontsize=24)
        ax_c.set_ylabel('Sickled fraction (%)', fontsize=24)
        ax_c.legend(loc='upper left', fontsize=22)
        plt.subplots_adjust(left=0.09, bottom=0.09, right=0.98, top=0.97)
        cumul_png = os.path.join(output_dir, f"{video_name}_sickle_rate_cumulative.png")
        plt.savefig(cumul_png, dpi=150)
        plt.close()
        print(f"  ✓ Cumulative curve saved: {video_name}_sickle_rate_cumulative.png")
    except Exception as e:
        print(f"  ⚠️ Cumulative curve generation failed: {e}")

    print(f"\nGenerated files:")
    print(f"  - {video_name}.mp4 (annotated video)")
    print(f"  - video_{video_name}_records.csv (detailed records)")
    print(f"  - state_ratio_plot_binary.png (overall sickling fraction)")
    print(f"  - frame0_class_pie.png (frame-0 class distribution)")
    print(f"  - {video_name}_sickle_rate_cumulative.png (cumulative rate curve)")

    print(f"\n✓ Video processing completed successfully!")
    print(f"✓ Output saved to: {output_dir}")
    return os.path.join(output_dir, f"{processed_video_stem}_records.csv")


if __name__ == '__main__':
    import os
    import datetime
    import argparse

    parser = argparse.ArgumentParser(
        description='Sickle Cell Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Primary input/output — same convention as sicklesight pipelines
    parser.add_argument('-i', '--inputs', type=str, default=None,
                        help='Comma-separated input video files, e.g. v1.mp4,v2.mp4')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Root output directory')
    # Alternative input mode
    parser.add_argument('--input-folder', type=str, default=None,
                        help='Folder to scan for .mp4 files (alternative to -i)')
    # Analysis options
    parser.add_argument('--seg', action='store_true',
                        help='Enable seg model for aspect-ratio ISC reclassification')
    parser.add_argument('--aspect-ratio', type=float, default=None,
                        help='ISC aspect ratio threshold (default 1.6)')
    parser.add_argument('--max-time', type=float, default=None,
                        help='Max duration to process per video in seconds (e.g. 120)')
    parser.add_argument('--max-frame', type=int, default=None,
                        help='Max frames to process per video (default: 480); overridden by --max-time')
    parser.add_argument('--max-duration', type=float, default=None,
                        help='[deprecated] alias for --max-time')
    parser.add_argument('--det-conf', type=str, default='auto',
                        help='YOLO detection confidence: float (e.g. 0.5) or "auto" to '
                             'estimate from video density (default: auto)')
    args = parser.parse_args()

    if args.seg:
        Config.USE_SEG_RECLASSIFY = True
    if args.aspect_ratio is not None:
        Config.SEG_ASPECT_RATIO_THRESHOLD = args.aspect_ratio
    # Priority: --max-time (or alias --max-duration) > --max-frame > default 120s
    _max_time_s = args.max_time or args.max_duration
    if _max_time_s is not None:
        Config.MAX_TIME_S = float(_max_time_s)
        print(f"  Max duration: {Config.MAX_TIME_S}s (converted to frames per video at runtime)")
    elif args.max_frame is not None:
        Config.MAX_TIME_S = None
        Config.MAX_VIDEO_FRAMES = args.max_frame
        print(f"  Max frames: {Config.MAX_VIDEO_FRAMES}")
    else:
        # default: Config.MAX_TIME_S = 120.0 (set in Config class)
        print(f"  Max duration: {Config.MAX_TIME_S}s (default, converted to frames per video at runtime)")
    if args.det_conf == 'auto':
        Config.AUTO_DET_CONF = True
    else:
        try:
            Config.DETECTION_CONFIDENCE = float(args.det_conf)
            Config.AUTO_DET_CONF = False
            print(f"  DETECTION_CONFIDENCE manually set to {Config.DETECTION_CONFIDENCE}")
        except ValueError:
            print(f"Warning: --det-conf '{args.det_conf}' invalid; falling back to auto-detect.")
            Config.AUTO_DET_CONF = True

    # ========== Model paths ==========
    MODEL_PATHS = {
        'yolo':        os.path.join(CELLBOX_MODELS_DIR, 'yolo', 'best.pt'),
        'seg':         os.path.join(CELLBOX_MODELS_DIR, 'seg', 'best.pt'),
        'mc_base_dir': os.path.join(CELLBOX_MODELS_DIR, 'efficientnet'),
        'mc_prefix':   'fold',
        'mc_n_folds':  5,
        'siamese':     os.path.join(CELLBOX_MODELS_DIR, 'siamese', 'model.pth'),
    }
    video_list = [

    # "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/4videos_2patient/0%-2147-osivelotor-1.mp4",
    # "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/4videos_2patient/0%-2147-osivelotor-2.mp4",
    # "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/4videos_2patient/0%-2147-osivelotor-3.mp4",
    # "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/4videos_2patient/0%-2147-osivelotor-4.mp4"
    # "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/MovingVideos_2026506/MD11-01-0per-osi-1.mp4",
    # "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/MovingVideos_2026506/MD11-01-0per-osi-2.mp4",
    # "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/MovingVideos_2026506/MD11-01-0per-osi-3.mp4",
    # "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/MovingVideos_2026506/MD11-01-0per-osi-4.mp4"
    # "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/V1.mp4",
    # "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/V2.mp4",
    # "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/V3.mp4",
    # "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/V4.mp4"
    #"/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/4videos_2patient/0%-2147-osivelotor-1_cleaned.mp4",
    "/home/sqma/cell_cluster/HeLi/track_cell/SCD_track/Jianlu_sickle_cell/vedio/4videos_2patient/0%-2147-osivelotor-1_cleaned2.mp4"


    ]
    
    print(f"{'='*50}")
    print("Sickle Cell Analysis System")
    print(f"{'='*50}")
    print(f"  - Processing mode: Single video")
    print(f"  - Ret time window: {Config.CLASS_TIME_WINDOWS_SEC['Ret']}s (fps-adaptive)")
    print(f"  - Other classes time window: {Config.CLASS_TIME_WINDOWS_SEC['Dis']}s (fps-adaptive)")
    print(f"  - Cell classes: {', '.join(MC_CLASSES)}")
    print("=" * 50)
    
    # Display model paths
    print("\nModel paths:")
    print(f"  YOLO:    {MODEL_PATHS['yolo']}")
    print(f"  Seg:     {MODEL_PATHS['seg']}")
    mc_fold1 = os.path.join(MODEL_PATHS['mc_base_dir'], f"{MODEL_PATHS['mc_prefix']}1_best.pth")
    print(f"  MC(x{MODEL_PATHS['mc_n_folds']}): {MODEL_PATHS['mc_base_dir']}")
    print(f"  Siamese: {MODEL_PATHS['siamese']}")

    # Verify model files exist
    print("\nVerifying model files:")
    yolo_exists    = os.path.exists(MODEL_PATHS['yolo'])
    seg_exists     = os.path.exists(MODEL_PATHS['seg'])
    mc_fold1_exists = os.path.exists(mc_fold1)
    siamese_exists = os.path.exists(MODEL_PATHS['siamese'])

    print(f"  YOLO:    {'OK' if yolo_exists else 'MISSING'}")
    print(f"  Seg:     {'OK' if seg_exists else 'MISSING'}")
    print(f"  MC fold1:{'OK' if mc_fold1_exists else 'MISSING'}")
    print(f"  Siamese: {'OK' if siamese_exists else 'MISSING'}")

    if not all([yolo_exists, seg_exists, mc_fold1_exists, siamese_exists]):
        print(f"\nWARNING: Some model files are missing. Check MODEL_PATHS in main.py.")
    
    print("=" * 50)
    
    # ========== 视频路径配置 ==========
    # 优先级: -i (逗号分隔文件 或 单个文件夹) > --input-folder > 脚本内 video_list
    if args.inputs:
        _input = args.inputs.strip()
        if os.path.isdir(_input):
            # -i 传入的是文件夹
            video_list = sorted([
                os.path.join(_input, f) for f in os.listdir(_input)
                if f.lower().endswith(('.mp4', '.avi')) and '_jitter_cleaned' not in f
            ])
            if not video_list:
                print(f"WARNING: no video files found in {_input}")
                sys.exit(1)
        else:
            video_list = [v.strip() for v in _input.split(',') if v.strip()]
    elif args.input_folder:
        folder = args.input_folder
        video_list = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(('.mp4', '.avi')) and '_jitter_cleaned' not in f
        ])
        if not video_list:
            print(f"WARNING: no video files found in {folder}")
            sys.exit(1)
    else:
        video_list = []

    print(f"Processing {len(video_list)} video(s):")
    for i, video_path in enumerate(video_list, 1):
        print(f"  Video {i}: {os.path.basename(video_path)}")
    print()
    
    records_csv_list = []
    for i, video_path in enumerate(video_list, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(video_list)}] Processing: {os.path.basename(video_path)}")
        print(f"{'='*80}")

        try:
            records_csv = main(video_path, MODEL_PATHS, output_root=args.output_dir)
            if records_csv:
                records_csv_list.append(records_csv)
            print(f"  Done: {os.path.basename(video_path)}")
        except Exception as e:
            print(f"  FAILED: {os.path.basename(video_path)}")
            print(f"  Error: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"All {len(video_list)} video(s) processed.")
    print(f"{'='*80}")

    if len(records_csv_list) > 1:
        print(f"\n{'='*80}")
        print("Starting combined multi-video analysis...")
        print(f"{'='*80}")

        try:
            # Read fps from the first video for time_s conversion
            _cap = cv2.VideoCapture(video_list[0])
            _fps = _cap.get(cv2.CAP_PROP_FPS) or 4.0
            _cap.release()
            combine_multi_video_csv_data(records_csv_list, fps=_fps, output_root=args.output_dir)
            print("Combined analysis complete.")
        except Exception as e:
            print(f"Combined analysis failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nOnly one video, skipping combined analysis.")
