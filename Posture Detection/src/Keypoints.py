import cv2
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from config import (
    INPUT_VIDEO, KEYPOINTS_JSON, YOLO_MODEL,
    KEYPOINT_NAMES, KEYPOINT_CONF_THRESHOLD,
    EXTRACT_EVERY_N_FRAMES, KEYPOINTS_DIR
)


def build_person_keypoints(kp_xy, kp_conf, person_idx):

    person_kps = {}

    for kp_idx, kp_name in KEYPOINT_NAMES.items():
        x    = float(kp_xy[person_idx][kp_idx][0])
        y    = float(kp_xy[person_idx][kp_idx][1])
        conf = float(kp_conf[person_idx][kp_idx])

        if conf < KEYPOINT_CONF_THRESHOLD:
            person_kps[kp_name] = {
                "x":        None,
                "y":        None,
                "conf":     conf,
                "occluded": True
            }
        else:
            person_kps[kp_name] = {
                "x":        x,
                "y":        y,
                "conf":     conf,
                "occluded": False
            }

    return person_kps


def is_person_usable(person_kps):
    left_ok  = not person_kps["left_shoulder"]["occluded"]
    right_ok = not person_kps["right_shoulder"]["occluded"]
    return left_ok or right_ok


def extract_keypoints():
    if not os.path.exists(INPUT_VIDEO):
        print(f"[ERROR] Video not found: {INPUT_VIDEO}")
        print(f"        Please place your classroom video at that path.")
        sys.exit(1)

    os.makedirs(KEYPOINTS_DIR, exist_ok=True)

    print(f"[INFO] Loading YOLO model: {YOLO_MODEL}")
    model = YOLO(YOLO_MODEL)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video opened. Total frames: {total_frames}")
    print(f"[INFO] Sampling every {EXTRACT_EVERY_N_FRAMES} frames.")
    print("[INFO] Starting extraction — this may take several minutes on CPU...\n")
    all_frames_data = []
    frame_idx       = 0
    saved_count     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % EXTRACT_EVERY_N_FRAMES != 0:
            frame_idx += 1
            continue

        results = model(frame, verbose=False)

        frame_data = {
            "frame_idx": frame_idx,
            "persons":   []
        }

        for result in results:
            if result.keypoints is None:
                continue

            kp_xy   = result.keypoints.xy.cpu().numpy()
            kp_conf = result.keypoints.conf.cpu().numpy()
            boxes   = result.boxes.xyxy.cpu().numpy()

            for pid in range(len(kp_xy)):
                person_kps = build_person_keypoints(kp_xy, kp_conf, pid)

                if not is_person_usable(person_kps):
                    continue

                frame_data["persons"].append({
                    "person_id": pid,
                    "bbox":      boxes[pid].tolist(),
                    "keypoints": person_kps
                })

        all_frames_data.append(frame_data)
        saved_count += 1

        if frame_idx % 150 == 0:
            pct = (frame_idx / total_frames) * 100
            print(f"  Frame {frame_idx}/{total_frames}  ({pct:.1f}%)")

        frame_idx += 1

    cap.release()

    with open(KEYPOINTS_JSON, "w") as f:
        json.dump(all_frames_data, f, indent=2)

    print(f"\n[DONE] Processed {saved_count} frames.")
    print(f"[DONE] Keypoints saved to: {KEYPOINTS_JSON}")


if __name__ == "__main__":
    extract_keypoints()
