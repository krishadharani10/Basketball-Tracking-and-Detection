from pathlib import Path
import urllib.request
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

VIDEO_FILE = BASE_DIR / "/Users/as/ku/sem_7/CV/MiniProject/basketball_vid.mp4"

OUTPUT_DIR = BASE_DIR / "basketball_output"

PLAYER_MODEL_PATH = BASE_DIR / "yolo11n.pt"
BALL_MODEL_PATH = BASE_DIR / "BODD_yolov8n_0001.pt"

PLAYER_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
)
BALL_MODEL_URL = (
    "https://huggingface.co/GabrieleGiudici/E-BARD-detection-models/"
    "resolve/main/BODD_yolov8n_0001.pt"
)

PLAYER_CONFIDENCE = 0.25
PLAYER_IMAGE_SIZE = 960
BALL_IMAGE_SIZE = 960

MIN_TRACK_DETECTIONS = 15
MAX_TRACKS_IN_GRAPHS = 12


BALL_CONFIDENCE = 0.05     
BALL_CONF_FLOOR = 0.15      

REJECT_HEADS = True          
HEAD_ZONE_HEIGHT = 0.20      
HEAD_ZONE_WIDTH = 0.70      

BALL_SIZE_RANGE = (0.04, 0.30)  
BALL_ASPECT_RANGE = (0.5, 2.0)  
STATIC_FRACTION = 0.5           

MAX_BALL_SPEED = 0.05        
MAX_LINK_FRAMES = 8          
CHAIN_BONUS = 0.15           
MIN_CHAIN_DETECTIONS = 4     
MAX_INTERPOLATE_FRAMES = 8   

DRAW_TRAIL = True          
BALL_TRAIL_LENGTH = 12     
TRAIL_MAX_GAP = 4          
DRAW_PLAYERS = True        
DEBUG_SHOW_REJECTED = False 

EVALUATION_IOU_THRESHOLD = 0.50
MAP_IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)


def download_model_if_missing(model_path, model_url):
    if model_path.exists() and model_path.stat().st_size > 0:
        print(f"Found model: {model_path.name}")
        return

    print(f"Downloading {model_path.name} ...")
    try:
        urllib.request.urlretrieve(model_url, str(model_path))
    except Exception as error:
        raise RuntimeError(
            f"Could not download {model_path.name}.\n"
            f"You can download it manually from:\n{model_url}"
        ) from error

    if not model_path.exists() or model_path.stat().st_size == 0:
        raise RuntimeError(f"Model download failed: {model_path}")


def get_model_names(model):
    names = model.names
    if isinstance(names, dict):
        return names
    return {index: name for index, name in enumerate(names)}


def get_ball_candidates(result, model_names):
    """Every ball-class box in one frame (no choosing yet)."""
    if result.boxes is None or len(result.boxes) == 0:
        return []

    ball_class_ids = [
        class_id for class_id, class_name in model_names.items()
        if "ball" in str(class_name).lower()
    ]

    candidates = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        if ball_class_ids and class_id not in ball_class_ids:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        candidates.append({
            "bbox": (x1, y1, x2, y2),
            "conf": float(box.conf[0].item()),
            "cx": (x1 + x2) / 2,
            "cy": (y1 + y2) / 2,
            "w": w,
            "h": h,
        })
    return candidates


def in_head_zone(cand, persons):
    for _, px1, py1, px2, py2 in persons:
        pw, ph = px2 - px1, py2 - py1
        margin = (1 - HEAD_ZONE_WIDTH) / 2 * pw
        if (px1 + margin <= cand["cx"] <= px2 - margin
                and py1 - 0.05 * ph <= cand["cy"] <= py1 + HEAD_ZONE_HEIGHT * ph):
            return True
    return False


def size_is_plausible(cand, persons):
    """Ball width should be a sensible fraction of the nearest player's height."""
    nearest = min(
        persons,
        key=lambda p: np.hypot(cand["cx"] - (p[1] + p[3]) / 2,
                               cand["cy"] - (p[2] + p[4]) / 2)
    )
    person_height = max(1, nearest[4] - nearest[2])
    ratio = cand["w"] / person_height
    return BALL_SIZE_RANGE[0] <= ratio <= BALL_SIZE_RANGE[1]


def expand_persons(persons_per_frame, hold=6):
    """
    If the player detector misses someone for a few frames, keep using their last
    box, so a head that is briefly 'unprotected' is still recognised as a head.
    """
    last_seen = {}                  
    expanded = []
    for t, persons in enumerate(persons_per_frame):
        current = {p[0] for p in persons}
        for p in persons:
            last_seen[p[0]] = (t, p)
        extra = [p for tid, (ft, p) in last_seen.items()
                 if tid not in current and t - ft <= hold]
        expanded.append(list(persons) + extra)
    return expanded


def filter_candidates(cands_per_frame, persons_per_frame, frame_width):
    """Rule-based removal of detections that cannot be the ball."""
    n = len(cands_per_frame)
    cell = max(12, frame_width // 80)

    counts = {}
    for cands in cands_per_frame:
        for key in {(int(c["cx"] // cell), int(c["cy"] // cell)) for c in cands}:
            counts[key] = counts.get(key, 0) + 1
    static_limit = max(15, STATIC_FRACTION * n)

    def is_static(c):
        gx, gy = int(c["cx"] // cell), int(c["cy"] // cell)
        total = sum(counts.get((gx + i, gy + j), 0)
                    for i in (-1, 0, 1) for j in (-1, 0, 1))
        return total >= static_limit

    kept_per_frame, rejected_per_frame = [], []
    stats = {"shape": 0, "static": 0, "head": 0, "size": 0}

    for t, cands in enumerate(cands_per_frame):
        persons = persons_per_frame[t]
        kept, rejected = [], []
        for c in cands:
            aspect = c["w"] / c["h"]
            if not (BALL_ASPECT_RANGE[0] <= aspect <= BALL_ASPECT_RANGE[1]):
                reason = "shape"
            elif is_static(c):
                reason = "static"
            elif persons and REJECT_HEADS and in_head_zone(c, persons):
                reason = "head"
            elif persons and not size_is_plausible(c, persons):
                reason = "size"
            else:
                kept.append(c)
                continue
            stats[reason] += 1
            rejected.append(c)
        kept_per_frame.append(kept)
        rejected_per_frame.append(rejected)

    return kept_per_frame, rejected_per_frame, stats


def select_ball_path(kept_per_frame, frame_width):
    n = len(kept_per_frame)
    step_limit = MAX_BALL_SPEED * frame_width
    restart_penalty = 0.2

    score = [[] for _ in range(n)]
    back = [[] for _ in range(n)]
    pref_best = (0.0, None)       
    best_overall = (0.0, None)

    for t in range(n):
        old = t - MAX_LINK_FRAMES - 1
        if old >= 0:
            for k, s in enumerate(score[old]):
                if s > pref_best[0]:
                    pref_best = (s, (old, k))

        for k, c in enumerate(kept_per_frame[t]):
            value = c["conf"] - BALL_CONF_FLOOR
            best_s, best_b = value, None

            if pref_best[1] is not None:
                s = pref_best[0] - restart_penalty + value
                if s > best_s:
                    best_s, best_b = s, pref_best[1]

            for gap in range(1, MAX_LINK_FRAMES + 1):
                tp = t - gap
                if tp < 0:
                    break
                limit = step_limit * gap
                for kp, cp in enumerate(kept_per_frame[tp]):
                    d = np.hypot(c["cx"] - cp["cx"], c["cy"] - cp["cy"])
                    if d <= limit:
                        penalty = 0.3 * (d / limit) ** 2
                        if gap <= 3 and d <= 0.5 * limit:
                            penalty -= CHAIN_BONUS
                    else:
                        penalty = 1.0 + (d - limit) / limit
                    s = score[tp][kp] - penalty + value
                    if s > best_s:
                        best_s, best_b = s, (tp, kp)

            score[t].append(best_s)
            back[t].append(best_b)
            if best_s > best_overall[0]:
                best_overall = (best_s, (t, k))

    chosen = {}
    node = best_overall[1]
    while node is not None:
        chosen[node[0]] = kept_per_frame[node[0]][node[1]]
        node = back[node[0]][node[1]]

    frames = sorted(chosen)
    chains, current = [], []
    for f in frames:
        if current and f - current[-1] > MAX_LINK_FRAMES:
            chains.append(current)
            current = []
        current.append(f)
    if current:
        chains.append(current)
    for chain in chains:
        if len(chain) < MIN_CHAIN_DETECTIONS:
            for f in chain:
                del chosen[f]

    return chosen


def build_ball_arrays(chosen, n):
    """Per-frame ball centre; short gaps filled with a straight line."""
    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    detected = np.zeros(n, dtype=bool)

    for f, c in chosen.items():
        cx[f], cy[f] = c["cx"], c["cy"]
        detected[f] = True

    frames = sorted(chosen)
    for a, b in zip(frames[:-1], frames[1:]):
        if 1 < b - a <= MAX_INTERPOLATE_FRAMES + 1:
            t = np.arange(a + 1, b)
            cx[t] = np.interp(t, [a, b], [cx[a], cx[b]])
            cy[t] = np.interp(t, [a, b], [cy[a], cy[b]])

    return cx, cy, detected

def create_speed_graph(data, time_column, speed_column,
                       track_column, title, output_path,
                       max_tracks=None):
    if data.empty:
        print(f"No data available for {output_path.name}")
        return

    plt.figure(figsize=(13, 7))
    grouped = list(data.groupby(track_column))

    if max_tracks is not None:
        grouped = sorted(
            grouped, key=lambda item: len(item[1]), reverse=True
        )[:max_tracks]

    for track_id, group in grouped:
        group = group.sort_values(time_column)
        plt.plot(
            group[time_column], group[speed_column],
            marker=".", markersize=3, linewidth=1.3,
            label=f"ID {track_id}"
        )

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Speed (pixels/second)")
    plt.grid(True, alpha=0.3)

    if len(grouped) <= 15:
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def create_trajectory_plot(data, title, output_path,
                           max_tracks=None, separate_tracks=False):
    if data.empty:
        print(f"No data available for {output_path.name}")
        return

    plt.figure(figsize=(12, 7))

    if separate_tracks:
        counts = data["track_id"].value_counts()
        selected_ids = counts.head(max_tracks or len(counts)).index

        for track_id in selected_ids:
            group = data[data["track_id"] == track_id].sort_values("frame")
            plt.plot(
                group["center_x"], group["center_y"],
                marker=".", markersize=2, linewidth=1.2,
                label=f"ID {track_id}"
            )
        plt.legend()
    else:
        ordered = data.sort_values("frame")

        segment_id = (ordered["frame"].diff() > MAX_LINK_FRAMES).cumsum()
        for _, segment in ordered.groupby(segment_id):
            plt.plot(
                segment["center_x"], segment["center_y"],
                marker=".", markersize=3, linewidth=1.2, color="tab:orange"
            )

    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def create_heatmap(data, frame_width, frame_height, title, output_path):
    if data.empty:
        print(f"No data available for {output_path.name}")
        return

    x_values = data["center_x"].astype(float).to_numpy()
    y_values = data["center_y"].astype(float).to_numpy()

    heatmap, _, _ = np.histogram2d(
        y_values, x_values,
        bins=[
            max(10, frame_height // 20),
            max(10, frame_width // 20)
        ],
        range=[[0, frame_height], [0, frame_width]]
    )

    plt.figure(figsize=(12, 7))
    plt.imshow(
        heatmap, cmap="hot", interpolation="nearest",
        extent=[0, frame_width, frame_height, 0], aspect="auto"
    )
    plt.colorbar(label="Number of tracked positions")
    plt.title(title)
    plt.xlabel("X position (pixels)")
    plt.ylabel("Y position (pixels)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def calculate_iou(box_a, box_b):
    """Calculate Intersection over Union for two boxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)
    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def average_precision_101_point(recall_values, precision_values):
    """COCO-style 101-point interpolated AP."""
    if len(recall_values) == 0:
        return np.nan

    recall_values = np.asarray(recall_values, dtype=float)
    precision_values = np.asarray(precision_values, dtype=float)
    recall_levels = np.linspace(0.0, 1.0, 101)

    interpolated = []
    for recall_level in recall_levels:
        eligible = precision_values[recall_values >= recall_level]
        interpolated.append(float(np.max(eligible)) if len(eligible) else 0.0)

    return float(np.mean(interpolated))


def match_detections_for_threshold(predictions, ground_truth, iou_threshold):
    """
    Match confidence-ranked predictions to ground truth.
    A ground-truth box can be matched only once per frame and class.
    """
    matched_ground_truth = set()
    true_positives = []
    false_positives = []
    matched_ious = []

    predictions = predictions.sort_values(
        "confidence", ascending=False
    ).reset_index(drop=True)

    for _, prediction in predictions.iterrows():
        same_frame_gt = ground_truth[
            (ground_truth["frame"] == prediction["frame"]) &
            (ground_truth["object_type"] == prediction["object_type"])
        ]

        best_iou = 0.0
        best_gt_index = None

        for gt_index, gt in same_frame_gt.iterrows():
            key = (int(gt["frame"]), str(gt["object_type"]), int(gt_index))
            if key in matched_ground_truth:
                continue

            iou = calculate_iou(
                [prediction["x1"], prediction["y1"],
                 prediction["x2"], prediction["y2"]],
                [gt["x1"], gt["y1"], gt["x2"], gt["y2"]]
            )

            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            gt = ground_truth.loc[best_gt_index]
            matched_ground_truth.add((
                int(gt["frame"]), str(gt["object_type"]), int(best_gt_index)
            ))
            true_positives.append(1)
            false_positives.append(0)
            matched_ious.append(best_iou)
        else:
            true_positives.append(0)
            false_positives.append(1)

    return (
        np.asarray(true_positives, dtype=float),
        np.asarray(false_positives, dtype=float),
        matched_ious
    )


def evaluate_one_class(predictions, ground_truth, object_type):
    class_predictions = predictions[
        predictions["object_type"] == object_type
    ].copy()
    class_gt = ground_truth[
        ground_truth["object_type"] == object_type
    ].copy()

    number_gt = len(class_gt)
    if number_gt == 0:
        return None

    tp, fp, ious = match_detections_for_threshold(
        class_predictions, class_gt, EVALUATION_IOU_THRESHOLD
    )

    total_tp = int(tp.sum())
    total_fp = int(fp.sum())
    false_negatives = max(0, number_gt - total_tp)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / number_gt if number_gt else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    ap_by_threshold = {}
    for threshold in MAP_IOU_THRESHOLDS:
        tp_curve, fp_curve, _ = match_detections_for_threshold(
            class_predictions, class_gt, float(threshold)
        )

        if len(tp_curve) == 0:
            ap = 0.0
        else:
            cumulative_tp = np.cumsum(tp_curve)
            cumulative_fp = np.cumsum(fp_curve)
            precision_curve = cumulative_tp / np.maximum(
                cumulative_tp + cumulative_fp, 1e-12
            )
            recall_curve = cumulative_tp / number_gt
            ap = average_precision_101_point(
                recall_curve, precision_curve
            )
        ap_by_threshold[f"AP@{threshold:.2f}"] = ap

    map_50 = ap_by_threshold["AP@0.50"]
    map_50_95 = float(np.mean(list(ap_by_threshold.values())))

    result = {
        "object_type": object_type,
        "ground_truth_objects": number_gt,
        "predicted_objects": len(class_predictions),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": false_negatives,
        "precision_at_iou_0_50": precision,
        "recall_at_iou_0_50": recall,
        "mean_iou_of_true_positives": mean_iou,
        "AP_at_0_50": map_50,
        "mAP_at_0_50_to_0_95": map_50_95
    }
    result.update(ap_by_threshold)
    return result


def run_evaluation(output_dir):
    gt_path = output_dir / "ground_truth.csv"
    frames_path = output_dir / "ground_truth_frames.csv"
    predictions_path = output_dir / "all_object_detections.csv"

    if not gt_path.exists():
        pd.DataFrame(columns=[
            "frame", "object_type", "x1", "y1", "x2", "y2"
        ]).to_csv(gt_path, index=False)

    if not frames_path.exists():
        pd.DataFrame(columns=["frame", "annotated"]).to_csv(
            frames_path, index=False
        )

    ground_truth = pd.read_csv(gt_path)
    annotated_frames = pd.read_csv(frames_path)

    if ground_truth.empty or annotated_frames.empty:
        message = (
            "Evaluation not calculated: ground-truth annotations are missing. "
            "Fill ground_truth.csv with manually labelled boxes and "
            "ground_truth_frames.csv with every frame you reviewed."
        )
        (output_dir / "evaluation_metrics.txt").write_text(
            message, encoding="utf-8"
        )
        pd.DataFrame([{
            "status": "NOT CALCULATED",
            "reason": "Manual ground-truth annotations are required."
        }]).to_csv(output_dir / "evaluation_metrics.csv", index=False)
        print("\n" + message)
        return

    required_gt_columns = {
        "frame", "object_type", "x1", "y1", "x2", "y2"
    }
    required_frame_columns = {"frame", "annotated"}

    if not required_gt_columns.issubset(ground_truth.columns):
        raise ValueError(
            "ground_truth.csv must contain: frame, object_type, x1, y1, x2, y2"
        )
    if not required_frame_columns.issubset(annotated_frames.columns):
        raise ValueError(
            "ground_truth_frames.csv must contain: frame, annotated"
        )

    ground_truth["frame"] = pd.to_numeric(
        ground_truth["frame"], errors="coerce"
    )
    annotated_frames["frame"] = pd.to_numeric(
        annotated_frames["frame"], errors="coerce"
    )
    annotated_frames["annotated"] = pd.to_numeric(
        annotated_frames["annotated"], errors="coerce"
    )

    ground_truth = ground_truth.dropna(
        subset=["frame", "object_type", "x1", "y1", "x2", "y2"]
    ).copy()
    annotated_frames = annotated_frames[
        annotated_frames["annotated"] == 1
    ].dropna(subset=["frame"]).copy()

    if annotated_frames.empty:
        raise ValueError(
            "No reviewed frames found. Add frame numbers with annotated=1."
        )

    ground_truth["frame"] = ground_truth["frame"].astype(int)
    annotated_frames["frame"] = annotated_frames["frame"].astype(int)
    ground_truth["object_type"] = (
        ground_truth["object_type"].astype(str).str.lower().str.strip()
    )

    valid_types = {"person", "basketball"}
    invalid_types = set(ground_truth["object_type"]) - valid_types
    if invalid_types:
        raise ValueError(
            f"Unknown object_type values: {invalid_types}. "
            "Use 'person' or 'basketball'."
        )

    numeric_columns = ["x1", "y1", "x2", "y2"]
    for column in numeric_columns:
        ground_truth[column] = pd.to_numeric(
            ground_truth[column], errors="coerce"
        )
    ground_truth = ground_truth.dropna(subset=numeric_columns)

    reviewed_frames = set(annotated_frames["frame"].tolist())
    ground_truth = ground_truth[
        ground_truth["frame"].isin(reviewed_frames)
    ].copy()

    predictions = pd.read_csv(predictions_path)
    if predictions.empty:
        raise ValueError("No predictions found in all_object_detections.csv")

    predictions["frame"] = pd.to_numeric(predictions["frame"], errors="coerce")
    predictions["object_type"] = (
        predictions["object_type"].astype(str).str.lower().str.strip()
    )
    for column in ["x1", "y1", "x2", "y2", "confidence"]:
        predictions[column] = pd.to_numeric(
            predictions[column], errors="coerce"
        )

    predictions = predictions.dropna(
        subset=["frame", "x1", "y1", "x2", "y2", "confidence"]
    ).copy()
    predictions["frame"] = predictions["frame"].astype(int)
    predictions = predictions[
        predictions["frame"].isin(reviewed_frames)
    ].copy()

    results = []
    for object_type in ["person", "basketball"]:
        result = evaluate_one_class(
            predictions, ground_truth, object_type
        )
        if result is not None:
            results.append(result)

    if not results:
        message = (
            "No ground-truth objects were found in the reviewed frames. "
            "Add labelled boxes to ground_truth.csv."
        )
        (output_dir / "evaluation_metrics.txt").write_text(
            message, encoding="utf-8"
        )
        print(message)
        return

    metrics_df = pd.DataFrame(results)

    # Macro average across evaluated object classes.
    metric_columns = [
        "precision_at_iou_0_50",
        "recall_at_iou_0_50",
        "mean_iou_of_true_positives",
        "AP_at_0_50",
        "mAP_at_0_50_to_0_95"
    ]
    overall = {"object_type": "OVERALL (macro average)"}
    for column in metric_columns:
        overall[column] = metrics_df[column].mean()
    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([overall])],
        ignore_index=True
    )

    metrics_df.to_csv(
        output_dir / "evaluation_metrics.csv", index=False
    )

    lines = [
        "OBJECT DETECTION EVALUATION RESULTS",
        "===================================",
        f"Reviewed frames: {len(reviewed_frames)}",
        f"IoU threshold for precision/recall: {EVALUATION_IOU_THRESHOLD:.2f}",
        "AP uses 101-point interpolated precision-recall.",
        "mAP@0.50: AP at IoU 0.50, averaged across evaluated classes.",
        "mAP@0.50:0.95: AP averaged over IoU thresholds 0.50 to 0.95.",
        "",
        metrics_df.to_string(index=False),
        "",
        "NOTE: Results are valid only for the manually annotated frames."
    ]
    (output_dir / "evaluation_metrics.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print("\nEVALUATION RESULTS")
    print(metrics_df.to_string(index=False))
    print("\nSaved evaluation_metrics.csv and evaluation_metrics.txt")


PSEUDO_GT_MODEL = "yolov8n.pt"

PSEUDO_GT_FRAME_STEP = 5

PSEUDO_GT_PERSON_CONF = 0.30
PSEUDO_GT_BALL_CONF = 0.10


def generate_pseudo_ground_truth(video_path, output_dir):

    print("\nGenerating automatic reference annotations...")

    reference_model = YOLO(PSEUDO_GT_MODEL)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video for reference detection.")

    ground_truth_records = []
    reviewed_frames = []

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        if frame_number % PSEUDO_GT_FRAME_STEP == 0:

            reviewed_frames.append({
                "frame": frame_number,
                "annotated": 1
            })

            results = reference_model.predict(
                source=frame,
                imgsz=960,
                conf=0.10,
                verbose=False
            )

            if results and results[0].boxes is not None:

                for box in results[0].boxes:

                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())

                    if class_id == 0:
                        object_type = "person"

                        if confidence < PSEUDO_GT_PERSON_CONF:
                            continue

                    elif class_id == 32:
                        object_type = "basketball"

                        if confidence < PSEUDO_GT_BALL_CONF:
                            continue

                    else:
                        continue

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    ground_truth_records.append({
                        "frame": frame_number,
                        "object_type": object_type,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

        frame_number += 1

        if frame_number % 100 == 0:
            print(
                f"Reference detection processed "
                f"{frame_number} frames"
            )

    cap.release()

    ground_truth_df = pd.DataFrame(
        ground_truth_records,
        columns=[
            "frame", "object_type",
            "x1", "y1", "x2", "y2"
        ]
    )

    ground_truth_df.to_csv(
        output_dir / "ground_truth.csv",
        index=False
    )

    reviewed_df = pd.DataFrame(
        reviewed_frames,
        columns=["frame", "annotated"]
    )

    reviewed_df.to_csv(
        output_dir / "ground_truth_frames.csv",
        index=False
    )

    print("\nAutomatic reference annotations completed.")
    print(f"Frames analysed: {len(reviewed_df)}")
    print(f"Reference boxes generated: {len(ground_truth_df)}")
    print(f"People: {(ground_truth_df['object_type'] == 'person').sum()}")
    print(
        "Basketballs: "
        f"{(ground_truth_df['object_type'] == 'basketball').sum()}"
    )
    print("Saved ground_truth.csv")
    print("Saved ground_truth_frames.csv")


def main():
    if not VIDEO_FILE.exists():
        raise FileNotFoundError(
            f"Could not find video: {VIDEO_FILE}\n"
            "Place basketball_vid.mp4 in the same folder as main.py, "
            "or update VIDEO_FILE near the top of the script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    download_model_if_missing(PLAYER_MODEL_PATH, PLAYER_MODEL_URL)
    download_model_if_missing(BALL_MODEL_PATH, BALL_MODEL_URL)

    print("\nLoading YOLO models...")
    player_model = YOLO(str(PLAYER_MODEL_PATH))
    ball_model = YOLO(str(BALL_MODEL_PATH))
    ball_model_names = get_model_names(ball_model)
    print("Basketball model classes:", ball_model_names)

    cap = cv2.VideoCapture(str(VIDEO_FILE))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_FILE}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = OUTPUT_DIR / "basketball_tracked_video.mp4"
    video_writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height)
    )

    if not video_writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the output video.")

    player_records = []
    persons_per_frame = []   
    cands_per_frame = []    
    frame_number = 0

    print("\nPASS 1: detecting players and ball candidates...")
    print(f"Resolution: {frame_width} x {frame_height}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}\n")


    while True:
        success, frame = cap.read()
        if not success:
            break

        timestamp = frame_number / fps
        persons = []

        player_results = player_model.track(
            source=frame,
            persist=True,
            tracker="botsort.yaml",
            classes=[0],
            conf=PLAYER_CONFIDENCE,
            imgsz=PLAYER_IMAGE_SIZE,
            verbose=False
        )

        if player_results:
            player_result = player_results[0]
            if (
                player_result.boxes is not None
                and player_result.boxes.id is not None
            ):
                boxes = player_result.boxes
                for box, track_id_tensor in zip(boxes, boxes.id):
                    track_id = int(track_id_tensor.item())
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )
                    confidence = float(box.conf[0].item())

                    persons.append((track_id, x1, y1, x2, y2))
                    player_records.append({
                        "frame": frame_number,
                        "timestamp_seconds": round(timestamp, 3),
                        "object_type": "person",
                        "track_id": track_id,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "center_x": (x1 + x2) // 2,
                        "center_y": (y1 + y2) // 2,
                        "confidence": round(confidence, 4)
                    })

        ball_results = ball_model.predict(
            source=frame,
            conf=BALL_CONFIDENCE,
            imgsz=BALL_IMAGE_SIZE,
            verbose=False
        )
        candidates = (
            get_ball_candidates(ball_results[0], ball_model_names)
            if ball_results else []
        )

        persons_per_frame.append(persons)
        cands_per_frame.append(candidates)

        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Detected {frame_number}/{total_frames} frames")

    cap.release()


    print("\nCleaning ball detections...")

    total_candidates = sum(len(c) for c in cands_per_frame)
    kept_per_frame, rejected_per_frame, reject_stats = filter_candidates(
        cands_per_frame, expand_persons(persons_per_frame), frame_width
    )
    chosen = select_ball_path(kept_per_frame, frame_width)
    ball_x, ball_y, ball_detected = build_ball_arrays(chosen, frame_number)

    print(f"Ball candidates found by the model: {total_candidates}")
    print(f"  removed - on a head:     {reject_stats['head']}")
    print(f"  removed - static object: {reject_stats['static']}")
    print(f"  removed - wrong size:    {reject_stats['size']}")
    print(f"  removed - wrong shape:   {reject_stats['shape']}")
    print(f"Ball boxes kept in the final path: {len(chosen)}")
    if chosen:
        confs = sorted(c["conf"] for c in chosen.values())
        print(f"  confidence of kept boxes: min {confs[0]:.2f}, "
              f"median {confs[len(confs) // 2]:.2f}, max {confs[-1]:.2f}")

    print("\nPASS 2: writing the annotated video...")

    cap = cv2.VideoCapture(str(VIDEO_FILE))
    basketball_records = []
    ball_trail = []        

    for f in range(frame_number):
        success, frame = cap.read()
        if not success:
            break

        annotated_frame = frame.copy()
        timestamp = f / fps

        if DRAW_PLAYERS:
            for track_id, x1, y1, x2, y2 in persons_per_frame[f]:
                cv2.rectangle(
                    annotated_frame, (x1, y1), (x2, y2),
                    (0, 220, 0), 2
                )
                cv2.putText(
                    annotated_frame, f"Player ID {track_id}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 0), 2
                )
                cv2.circle(
                    annotated_frame, ((x1 + x2) // 2, (y1 + y2) // 2),
                    4, (255, 0, 0), -1
                )

        if DEBUG_SHOW_REJECTED:
            for c in rejected_per_frame[f]:
                x1, y1, x2, y2 = c["bbox"]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            for c in kept_per_frame[f]:
                if chosen.get(f) is not c:
                    x1, y1, x2, y2 = c["bbox"]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                                  (160, 160, 160), 1)

        has_position = not np.isnan(ball_x[f])

        if ball_detected[f]:
            c = chosen[f]
            x1, y1, x2, y2 = c["bbox"]
            center_x, center_y = int(ball_x[f]), int(ball_y[f])
            confidence = c["conf"]

            if ball_trail and f - ball_trail[-1][0] > TRAIL_MAX_GAP:
                ball_trail = []
            ball_trail.append((f, center_x, center_y))
            ball_trail = ball_trail[-BALL_TRAIL_LENGTH:]

            cv2.rectangle(
                annotated_frame, (x1, y1), (x2, y2),
                (0, 140, 255), 3
            )
            cv2.putText(
                annotated_frame, f"Basketball ({confidence:.2f})",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 140, 255), 2
            )

            if DRAW_TRAIL:
                for index in range(1, len(ball_trail)):
                    cv2.line(
                        annotated_frame,
                        ball_trail[index - 1][1:], ball_trail[index][1:],
                        (0, 255, 255), 2
                    )

            basketball_records.append({
                "frame": f,
                "timestamp_seconds": round(timestamp, 3),
                "object_type": "basketball",
                "detected": 1,
                "interpolated": 0,
                "track_id": 1,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "center_x": center_x, "center_y": center_y,
                "confidence": round(confidence, 4)
            })

        else:
            if has_position:
                cv2.circle(
                    annotated_frame, (int(ball_x[f]), int(ball_y[f])),
                    14, (0, 140, 255), 2
                )

            basketball_records.append({
                "frame": f,
                "timestamp_seconds": round(timestamp, 3),
                "object_type": "basketball",
                "detected": 0,
                "interpolated": int(has_position),
                "track_id": 1,
                "x1": np.nan, "y1": np.nan, "x2": np.nan, "y2": np.nan,
                "center_x": round(float(ball_x[f]), 1) if has_position else np.nan,
                "center_y": round(float(ball_y[f]), 1) if has_position else np.nan,
                "confidence": np.nan
            })

        cv2.putText(
            annotated_frame, f"Frame: {f}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        video_writer.write(annotated_frame)

        if (f + 1) % 100 == 0:
            print(f"Drawn {f + 1}/{frame_number} frames")

    cap.release()
    video_writer.release()
    print("\nVideo processing completed. Creating CSV files and graphs...")


    players_df = pd.DataFrame(player_records)
    basketball_df = pd.DataFrame(basketball_records)

    players_df.to_csv(
        OUTPUT_DIR / "player_detections.csv", index=False
    )
    basketball_df.to_csv(
        OUTPUT_DIR / "basketball_positions.csv", index=False
    )

    combined_records = []
    for row in player_records:
        combined_records.append({
            key: row[key] for key in [
                "frame", "timestamp_seconds", "object_type", "track_id",
                "x1", "y1", "x2", "y2", "center_x", "center_y", "confidence"
            ]
        })

    for row in basketball_records:
        if row["detected"] == 1:
            combined_records.append({
                key: row[key] for key in [
                    "frame", "timestamp_seconds", "object_type", "track_id",
                    "x1", "y1", "x2", "y2", "center_x", "center_y",
                    "confidence"
                ]
            })

    combined_df = pd.DataFrame(combined_records)
    combined_df.to_csv(
        OUTPUT_DIR / "all_object_detections.csv", index=False
    )

    if not players_df.empty:
        players_df = players_df.sort_values(
            ["track_id", "frame"]
        ).copy()

        players_df["delta_x"] = (
            players_df.groupby("track_id")["center_x"].diff()
        )
        players_df["delta_y"] = (
            players_df.groupby("track_id")["center_y"].diff()
        )
        players_df["delta_time"] = (
            players_df.groupby("track_id")["timestamp_seconds"].diff()
        )
        players_df["speed_pixels_per_second"] = (
            np.sqrt(players_df["delta_x"] ** 2 + players_df["delta_y"] ** 2)
            / players_df["delta_time"].replace(0, np.nan)
        )
        players_df.to_csv(
            OUTPUT_DIR / "player_detections_with_speed.csv", index=False
        )

        player_summary_df = (
            players_df.groupby("track_id")
            .agg(
                detections=("frame", "count"),
                first_frame=("frame", "min"),
                last_frame=("frame", "max"),
                average_speed_pixels_per_second=(
                    "speed_pixels_per_second", "mean"
                ),
                maximum_speed_pixels_per_second=(
                    "speed_pixels_per_second", "max"
                )
            )
            .reset_index()
        )
        player_summary_df.to_csv(
            OUTPUT_DIR / "player_track_summary.csv", index=False
        )

        eligible_ids = player_summary_df[
            player_summary_df["detections"] >= MIN_TRACK_DETECTIONS
        ]["track_id"].tolist()

        player_speed_data = players_df[
            players_df["track_id"].isin(eligible_ids)
        ].copy()

        if not player_speed_data.empty:
            create_speed_graph(
                player_speed_data,
                "timestamp_seconds",
                "speed_pixels_per_second",
                "track_id",
                "Player Movement Speed Over Time",
                OUTPUT_DIR / "player_speed_graph.png",
                MAX_TRACKS_IN_GRAPHS
            )

        create_trajectory_plot(
            players_df, "Player Trajectories",
            OUTPUT_DIR / "player_trajectory.png",
            MAX_TRACKS_IN_GRAPHS, separate_tracks=True
        )
        create_heatmap(
            players_df, frame_width, frame_height,
            "Player Position Heatmap",
            OUTPUT_DIR / "player_heatmap.png"
        )
    else:
        print("No player tracks were recorded.")
        pd.DataFrame(columns=[
            "track_id", "detections", "first_frame", "last_frame",
            "average_speed_pixels_per_second",
            "maximum_speed_pixels_per_second"
        ]).to_csv(OUTPUT_DIR / "player_track_summary.csv", index=False)

    detected_ball_df = basketball_df[
        basketball_df["detected"] == 1
    ].copy()

    if not detected_ball_df.empty:
        detected_ball_df = detected_ball_df.sort_values("frame")
        detected_ball_df["delta_x"] = detected_ball_df["center_x"].diff()
        detected_ball_df["delta_y"] = detected_ball_df["center_y"].diff()
        detected_ball_df["delta_time"] = (
            detected_ball_df["timestamp_seconds"].diff()
        )
        detected_ball_df["speed_pixels_per_second"] = (
            np.sqrt(
                detected_ball_df["delta_x"] ** 2
                + detected_ball_df["delta_y"] ** 2
            )
            / detected_ball_df["delta_time"].replace(0, np.nan)
        )
        detected_ball_df.to_csv(
            OUTPUT_DIR / "basketball_positions_with_speed.csv",
            index=False
        )

        create_speed_graph(
            detected_ball_df,
            "timestamp_seconds",
            "speed_pixels_per_second",
            "track_id",
            "Basketball Movement Speed Over Time",
            OUTPUT_DIR / "basketball_speed_graph.png"
        )
        create_trajectory_plot(
            detected_ball_df,
            "Basketball Trajectory",
            OUTPUT_DIR / "basketball_trajectory.png"
        )
        create_heatmap(
            detected_ball_df, frame_width, frame_height,
            "Basketball Position Heatmap",
            OUTPUT_DIR / "basketball_heatmap.png"
        )
    else:
        print("WARNING: Basketball was not detected in any frame.")


    ground_truth_path = OUTPUT_DIR / "ground_truth.csv"
    reviewed_frames_path = OUTPUT_DIR / "ground_truth_frames.csv"

    if not ground_truth_path.exists():
        generate_pseudo_ground_truth(VIDEO_FILE, OUTPUT_DIR)
        run_evaluation(OUTPUT_DIR)

    if not reviewed_frames_path.exists():
        pd.DataFrame(columns=["frame", "annotated"]).to_csv(
            reviewed_frames_path, index=False
        )
    run_evaluation(OUTPUT_DIR)
    total_ball_frames = len(basketball_df)
    detected_ball_frames = (
        int(basketball_df["detected"].sum())
        if not basketball_df.empty else 0
    )
    ball_detection_rate = (
        detected_ball_frames / total_ball_frames * 100
        if total_ball_frames > 0 else 0
    )
    unique_player_ids = (
        players_df["track_id"].nunique()
        if not players_df.empty else 0
    )

    summary = [
        "BASKETBALL OBJECT TRACKING PROJECT - RUN SUMMARY",
        "=================================================",
        f"Input video: {VIDEO_FILE.name}",
        f"Frames processed: {frame_number}",
        f"FPS: {fps:.2f}",
        f"Resolution: {frame_width} x {frame_height}",
        f"Unique player IDs: {unique_player_ids}",
        f"Basketball detected frames: {detected_ball_frames}/{total_ball_frames}",
        f"Basketball detection rate: {ball_detection_rate:.2f}%",
        f"Ball candidates found by the model: {total_candidates}",
        f"  removed as head: {reject_stats['head']}, static object: {reject_stats['static']}, "
        f"wrong size: {reject_stats['size']}, wrong shape: {reject_stats['shape']}",
        "",
        "IMPORTANT NOTES:",
        "- Ball detection rate is not precision, recall or accuracy.",
        "- Evaluation metrics require manually labelled ground truth.",
        "- Speed is measured in pixels/second, not km/h.",
        "- Real-world speed requires camera/court calibration.",
        "- Player IDs may change after occlusion or missed detections.",
        "- The ball is ONE path chosen from all candidates (heads, static objects and",
        "  impossible jumps are removed first); it is not multi-ball tracking.",
        "",
        "OUTPUT FILES:"
    ]

    for output_file in sorted(OUTPUT_DIR.iterdir()):
        if output_file.is_file():
            summary.append(f"- {output_file.name}")

    (OUTPUT_DIR / "run_summary.txt").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    print("\n========================================")
    print("          PROJECT COMPLETED")
    print("========================================")
    print(f"All outputs saved in: {OUTPUT_DIR}")
    print(f"Player IDs found: {unique_player_ids}")
    print(f"Basketball detection rate: {ball_detection_rate:.2f}%")
    print("Evaluation files: evaluation_metrics.csv and evaluation_metrics.txt")
    print("If evaluation says NOT CALCULATED, annotate frames and run again.")


if __name__ == "__main__":
    main()