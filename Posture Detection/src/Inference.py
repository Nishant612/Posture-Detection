import cv2
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    INPUT_VIDEO, OUTPUT_VIDEO, YOLO_MODEL,
    KEYPOINT_NAMES, KEYPOINT_CONF_THRESHOLD,
    CLASSIFIER_CONF_THRESHOLD, INFERENCE_EVERY_N_FRAMES,
    SMOOTH_WINDOW, LABEL_NAMES, LABEL_COLORS,
    NUM_CLASSES, FEATURE_SIZE, DROPOUT_RATE,
    CLASSIFIER_MODEL, OUTPUT_FPS, OUTPUT_DIR
)
from src.FeatureExtraction import extract_features
from src.Train import PostureClassifier


class TemporalSmoother:

    def __init__(self, window=SMOOTH_WINDOW):
        self.window  = window
        self.buffers = {}

    def update(self, person_id, prediction):
        if person_id not in self.buffers:
            self.buffers[person_id] = deque(maxlen=self.window)
        self.buffers[person_id].append(prediction)
        buf = list(self.buffers[person_id])
        return max(set(buf), key=buf.count)

    def reset(self):
        self.buffers.clear()


def draw_person(frame, bbox, label_idx, confidence, person_id):

    x1, y1, x2, y2 = map(int, bbox)
    color      = LABEL_COLORS[label_idx]
    label_text = f"P{person_id}: {LABEL_NAMES[label_idx]} ({confidence:.0%})"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (tw, th), _ = cv2.getTextSize(label_text,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

    cv2.putText(frame, label_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)


def draw_stats(frame, frame_idx, person_count, behaviour_counts):

    h, w = frame.shape[:2]
    panel_x = w - 220
    panel_y = 10

    lines = [
        f"Frame: {frame_idx}",
        f"People: {person_count}",
        "---",
    ] + [f"{LABEL_NAMES[k]}: {v}" for k, v in sorted(behaviour_counts.items())]

    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (panel_x, panel_y + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1)


def run_inference():

    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERROR] Video not found: {INPUT_VIDEO}")
        sys.exit(1)

    if not os.path.exists(CLASSIFIER_MODEL):
        print(f"[ERROR] Classifier not found: {CLASSIFIER_MODEL}")
        print("        Run train.py first.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from ultralytics import YOLO
    print(f"[INFO] Loading YOLO model: {YOLO_MODEL}")
    pose_model = YOLO(YOLO_MODEL)

    print(f"[INFO] Loading classifier: {CLASSIFIER_MODEL}")
    classifier = PostureClassifier()
    classifier.load_state_dict(torch.load(CLASSIFIER_MODEL, map_location="cpu"))
    classifier.eval()

    cap          = cv2.VideoCapture(INPUT_VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        OUTPUT_FPS,
        (frame_w, frame_h)
    )

    smoother   = TemporalSmoother()
    frame_idx  = 0
    last_frame = None

    print(f"[INFO] Running inference on {total_frames} frames...")
    print("[INFO] Press Q in the display window to stop early.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % INFERENCE_EVERY_N_FRAMES != 0:
            if last_frame is not None:
                out.write(last_frame)
            continue

        results          = pose_model(frame, verbose=False)
        behaviour_counts = {i: 0 for i in range(NUM_CLASSES)}
        person_count     = 0

        for result in results:
            if result.keypoints is None:
                continue

            kp_xy   = result.keypoints.xy.cpu().numpy()
            kp_conf = result.keypoints.conf.cpu().numpy()
            boxes   = result.boxes.xyxy.cpu().numpy()

            for pid in range(len(kp_xy)):

                kps = {}
                for kp_idx, kp_name in KEYPOINT_NAMES.items():
                    x    = float(kp_xy[pid][kp_idx][0])
                    y    = float(kp_xy[pid][kp_idx][1])
                    conf = float(kp_conf[pid][kp_idx])
                    kps[kp_name] = {
                        "x":        x    if conf >= KEYPOINT_CONF_THRESHOLD else None,
                        "y":        y    if conf >= KEYPOINT_CONF_THRESHOLD else None,
                        "conf":     conf,
                        "occluded": conf < KEYPOINT_CONF_THRESHOLD,
                    }

                features = extract_features(kps)
                if features is None:
                    continue

                with torch.no_grad():
                    feat_t  = torch.tensor([features], dtype=torch.float32)
                    logits  = classifier(feat_t)
                    probs   = torch.softmax(logits, dim=1)
                    conf_v  = probs.max().item()
                    pred    = probs.argmax(1).item()

                if conf_v < CLASSIFIER_CONF_THRESHOLD:
                    continue

                stable_pred = smoother.update(pid, pred)

                draw_person(frame, boxes[pid], stable_pred, conf_v, pid)
                behaviour_counts[stable_pred] += 1
                person_count += 1

        draw_stats(frame, frame_idx, person_count, behaviour_counts)

        out.write(frame)
        last_frame = frame.copy()

        cv2.imshow("Classroom Analysis  |  Q = quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] User stopped early.")
            break

        if frame_idx % 100 == 0:
            pct = (frame_idx / total_frames) * 100
            print(f"  Frame {frame_idx}/{total_frames}  ({pct:.1f}%)")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\n[DONE] Annotated video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    run_inference()
