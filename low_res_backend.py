import os

import cv2
import numpy as np
import torch
from skimage.measure import label, regionprops

from device_utils import get_torch_device, get_ultralytics_device

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CELLBOX_MODELS_DIR = os.path.join(SCRIPT_DIR, 'CellBox-Models')
DEFAULT_LOW_RES_YOLO_MODEL = os.path.join(CELLBOX_MODELS_DIR, 'yolo', 'best.pt')
DEFAULT_LOW_RES_SEG_MODEL = os.path.join(CELLBOX_MODELS_DIR, 'seg', 'best.pt')
DEFAULT_LOW_RES_TRACKER_CONFIG = os.path.join(CELLBOX_MODELS_DIR, 'configs', 'botsort_cell.yaml')

LOW_RES_REF_RESOLUTION = (5472, 3648)
LOW_RES_MIN_CELL_AREA = 30000
LOW_RES_EDGE_MARGIN = 3
LOW_RES_EDGE_BOX_AR_MAX = 1.4


def bbox_morphology(w, h):
    if min(w, h) <= 0:
        return 1.0, 0.5, 0.5
    return max(w, h) / min(w, h), 0.5, 0.5


def _eccentricity(mask):
    if mask.sum() == 0:
        return 0.0
    props = regionprops(label(mask))
    return props[0].eccentricity if props else 0.0


def _circularity(mask):
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


def compute_low_res_mask_morphology(crop, seg_model, conf_threshold=0.05, yolo_device=None):
    h, w = crop.shape[:2]
    fallback = bbox_morphology(w, h)
    if seg_model is None or crop.size == 0:
        return fallback

    try:
        results = seg_model.predict(
            source=crop,
            conf=conf_threshold,
            save=False,
            verbose=False,
            device=yolo_device,
        )
        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            return fallback

        masks = results[0].masks.data.cpu().numpy()
        mask_resized = cv2.resize(masks[0], (w, h))
        binary = (mask_resized > 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return fallback

        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) >= 5:
            _, axes, _ = cv2.fitEllipse(largest_contour)
            major_axis = max(axes)
            minor_axis = min(axes)
            ar_val = major_axis / minor_axis if minor_axis > 0 else fallback[0]
        else:
            rect = cv2.minAreaRect(largest_contour)
            width, height = rect[1]
            major_axis = max(width, height)
            minor_axis = min(width, height)
            ar_val = major_axis / minor_axis if minor_axis > 0 else fallback[0]

        return ar_val, _eccentricity(binary), _circularity(binary)
    except Exception as exc:
        print(f"Warning: Low-res mask morphology failed; using bbox fallback: {exc}")
        return fallback


def _reset_ultralytics_tracker(yolo_model):
    try:
        if hasattr(yolo_model, "predictor") and yolo_model.predictor is not None:
            if hasattr(yolo_model.predictor, "tracker") and yolo_model.predictor.tracker is not None:
                yolo_model.predictor.tracker.reset()
            elif hasattr(yolo_model.predictor, "trackers") and yolo_model.predictor.trackers:
                trackers = yolo_model.predictor.trackers
                tracker_iter = trackers.values() if isinstance(trackers, dict) else trackers
                for tracker in tracker_iter:
                    if hasattr(tracker, 'reset'):
                        tracker.reset()
        elif hasattr(yolo_model, "tracker") and yolo_model.tracker is not None:
            yolo_model.tracker.reset()
        yolo_model.predictor = None
    except Exception as exc:
        print(f"Warning: Could not reset YOLO tracker: {exc}")


def init_low_res_backend(yolo_model_path, tracker_config_path, seg_model_path=None, device=None):
    if YOLO is None:
        raise ImportError("ultralytics is required for --tracking_backend low_res.")
    if not os.path.exists(yolo_model_path):
        raise FileNotFoundError(f"Low-res YOLO model not found: {yolo_model_path}")
    if not os.path.exists(tracker_config_path):
        raise FileNotFoundError(f"Low-res tracker config not found: {tracker_config_path}")

    torch_device = device or get_torch_device()
    yolo_device = get_ultralytics_device(torch_device)

    yolo_model = YOLO(yolo_model_path)
    _reset_ultralytics_tracker(yolo_model)

    seg_model = None
    if seg_model_path and os.path.exists(seg_model_path):
        seg_model = YOLO(seg_model_path)
    elif seg_model_path:
        print(f"Warning: Low-res seg model not found; using bbox morphology fallback: {seg_model_path}")

    return {
        'yolo': yolo_model,
        'seg_model': seg_model,
        'yolo_device': yolo_device,
        'tracker_config': tracker_config_path,
        'invalid_ids': set(),
        'used_ids': set(),
        'next_id': 10000,
        'first_frame_ids': set(),
        'is_first_frame': True,
        'last_known_pos': {},
        'id_remap': {},
        'last_processed_frame': None,
        'max_jump_px': 200,
        'max_recovery_px': 200,
        'recovery_buffer': 60,
    }


def filter_low_res_boxes(results, img_w, img_h, min_cell_area=None,
                         margin=LOW_RES_EDGE_MARGIN,
                         edge_box_ar_max=LOW_RES_EDGE_BOX_AR_MAX):
    filtered, ids = [], []
    if min_cell_area is None:
        min_cell_area = LOW_RES_MIN_CELL_AREA
    if not hasattr(results, 'boxes') or results.boxes is None:
        return filtered, ids

    boxes = results.boxes
    ids_raw = (boxes.id.cpu().numpy()
               if (boxes.id is not None and len(boxes.id) > 0)
               else np.arange(len(boxes.xyxy)))

    for box, tid in zip(boxes.xyxy, ids_raw):
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        x1 = max(0, min(int(x1), img_w - 1))
        y1 = max(0, min(int(y1), img_h - 1))
        x2 = max(0, min(int(x2), img_w))
        y2 = max(0, min(int(y2), img_h))
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0 or w * h < min_cell_area:
            continue

        at_edge = (x1 < margin or y1 < margin or
                   x2 > img_w - margin or y2 > img_h - margin)
        if at_edge and max(w, h) / max(min(w, h), 1) > edge_box_ar_max:
            continue

        filtered.append((x1, y1, x2, y2))
        ids.append(int(tid))

    if len(filtered) <= 1:
        return filtered, ids

    try:
        import torchvision
        boxes_t = torch.tensor(filtered, dtype=torch.float32)
        areas = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        keep = torchvision.ops.nms(boxes_t, areas, iou_threshold=0.4).tolist()
        filtered = [filtered[i] for i in keep]
        ids = [ids[i] for i in keep]
    except Exception as exc:
        print(f"Warning: Low-res NMS failed; keeping pre-NMS boxes: {exc}")

    return filtered, ids


def auto_detect_low_res_conf(video_path, yolo_model, n_frames=5, yolo_device=None):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 1:
        cap.release()
        return 0.25

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
        print("  [low-res auto-conf] No detections in probe frames; using 0.25")
        return 0.25

    avg_frac = float(np.mean(fracs))
    if avg_frac >= 0.72:
        conf, density = 0.80, "sparse"
    elif avg_frac >= 0.62:
        conf, density = 0.50, "medium"
    else:
        conf, density = 0.25, "dense"

    print(f"  [low-res auto-conf] probed {len(fracs)} frames | "
          f"high-conf-frac={avg_frac:.3f} | density={density} | conf={conf}")
    return conf


def resolve_low_res_det_conf(video_path, yolo_model, det_conf, yolo_device=None):
    if isinstance(det_conf, str):
        if det_conf.lower() == 'auto':
            return auto_detect_low_res_conf(video_path, yolo_model, yolo_device=yolo_device)
        try:
            return float(det_conf)
        except ValueError:
            print(f"Warning: --low_res_det_conf '{det_conf}' invalid; using auto.")
            return auto_detect_low_res_conf(video_path, yolo_model, yolo_device=yolo_device)
    return float(det_conf)


def detect_low_res_frame(low_res_state, frame, frame_idx, det_conf=0.25, yolo_iou=0.6):
    img_h, img_w = frame.shape[:2]
    ref_w, ref_h = LOW_RES_REF_RESOLUTION
    res_scale = (img_w * img_h) / (ref_w * ref_h)
    adaptive_min_cell_area = max(100, int(LOW_RES_MIN_CELL_AREA * res_scale))

    res = low_res_state['yolo'].track(
        source=frame,
        persist=True,
        conf=det_conf,
        iou=yolo_iou,
        max_det=2000,
        tracker=low_res_state['tracker_config'],
        verbose=False,
        device=low_res_state.get('yolo_device'),
    )[0]

    boxes, ids = filter_low_res_boxes(
        res, img_w, img_h, min_cell_area=adaptive_min_cell_area)
    detections = []
    skipped = 0

    for (x1, y1, x2, y2), orig_id in zip(boxes, ids):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        if orig_id in low_res_state['invalid_ids']:
            while low_res_state['next_id'] in low_res_state['used_ids']:
                low_res_state['next_id'] += 1
            tid = low_res_state['next_id']
        else:
            tid = low_res_state['id_remap'].get(orig_id, orig_id)
        low_res_state['used_ids'].add(tid)

        if low_res_state['is_first_frame']:
            low_res_state['first_frame_ids'].add(tid)
            low_res_state['last_known_pos'][tid] = (cx, cy, frame_idx)
        else:
            if tid not in low_res_state['first_frame_ids']:
                recovered_tid = None
                best_dist = low_res_state['max_recovery_px']
                for known_tid, (lx, ly, lf) in low_res_state['last_known_pos'].items():
                    if known_tid not in low_res_state['first_frame_ids']:
                        continue
                    if frame_idx - lf > low_res_state['recovery_buffer']:
                        continue
                    dist = float(np.hypot(cx - lx, cy - ly))
                    if dist < best_dist:
                        best_dist = dist
                        recovered_tid = known_tid
                if recovered_tid is not None:
                    low_res_state['id_remap'][orig_id] = recovered_tid
                    tid = recovered_tid
                    print(f"    [low-res ID recovery] new {orig_id} -> {recovered_tid} "
                          f"(dist={best_dist:.1f}px, frame={frame_idx})")
                else:
                    skipped += 1
                    continue

            if tid in low_res_state['last_known_pos']:
                lx, ly, lf = low_res_state['last_known_pos'][tid]
                if lf == low_res_state['last_processed_frame']:
                    jump = float(np.hypot(cx - lx, cy - ly))
                    if jump > low_res_state['max_jump_px']:
                        print(f"    [low-res spatial jump] ID {tid} at frame {frame_idx} "
                              f"moved {jump:.1f}px; skipped")
                        skipped += 1
                        continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            low_res_state['invalid_ids'].add(orig_id)
            skipped += 1
            continue

        low_res_state['last_known_pos'][tid] = (cx, cy, frame_idx)
        detections.append({
            'track_id': tid,
            'bbox': (x1, y1, x2 - x1, y2 - y1),
            'crop': crop,
        })

    low_res_state['is_first_frame'] = False
    low_res_state['last_processed_frame'] = frame_idx
    if skipped:
        print(f"    Low-res backend skipped {skipped} boxes at frame {frame_idx}")

    return detections
