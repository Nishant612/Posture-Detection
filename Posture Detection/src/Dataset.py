import json
import os
import sys
import numpy as np
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    KEYPOINTS_JSON, LABELS_JSON,
    X_TRAIN_FILE, Y_TRAIN_FILE,
    LABEL_MAP, DATASET_DIR
)
from src.FeatureExtraction import extract_features


def build_dataset():
    for path in [KEYPOINTS_JSON, LABELS_JSON]:
        if not os.path.exists(path):
            print(f"[ERROR] File not found: {path}")
            sys.exit(1)

    os.makedirs(DATASET_DIR, exist_ok=True)

    print(f"[INFO] Loading keypoints from: {KEYPOINTS_JSON}")
    with open(KEYPOINTS_JSON) as f:
        keypoints_data = json.load(f)

    print(f"[INFO] Loading labels from:    {LABELS_JSON}")
    with open(LABELS_JSON) as f:
        labels_data = json.load(f)

    frame_label_map = {}
    for segment in labels_data:
        label_str = segment["label"]
        if label_str not in LABEL_MAP:
            print(f"[WARN] Unknown label '{label_str}' in labels.json — skipping.")
            continue
        label_int = LABEL_MAP[label_str]
        for fidx in range(segment["frame_start"], segment["frame_end"]):
            frame_label_map[fidx] = label_int

    print(f"[INFO] {len(frame_label_map)} frames have labels.")

    X = []
    y = []
    skipped_no_label    = 0
    skipped_no_features = 0

    for frame in keypoints_data:
        fidx      = frame["frame_idx"]
        label_int = frame_label_map.get(fidx)

        if label_int is None:
            skipped_no_label += 1
            continue

        for person in frame["persons"]:
            features = extract_features(person["keypoints"])

            if features is None:
                skipped_no_features += 1
                continue

            X.append(features)
            y.append(label_int)

    if len(X) == 0:
        print("[ERROR] No samples were generated.")
        print("        Make sure your labels.json frame ranges overlap")
        print("        with the frames in keypoints_data.json.")
        sys.exit(1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(X_TRAIN_FILE, X)
    np.save(Y_TRAIN_FILE, y)
    print(f"\n[DONE] Dataset built successfully.")
    print(f"       Total samples   : {len(X)}")
    print(f"       Feature size    : {X.shape[1]}")
    print(f"       Skipped (no label)   : {skipped_no_label} frames")
    print(f"       Skipped (bad person) : {skipped_no_features} persons")
    print()
    print("  Class distribution:")
    from config import LABEL_NAMES
    for cls, count in sorted(Counter(y.tolist()).items()):
        print(f"    {LABEL_NAMES[cls]:15s} : {count} samples")
    print()
    print(f"  Saved X → {X_TRAIN_FILE}")
    print(f"  Saved y → {Y_TRAIN_FILE}")


if __name__ == "__main__":
    build_dataset()
