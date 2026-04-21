import math
import json
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KEYPOINTS_JSON


def _get_xy(person_keypoints, name):
    kp = person_keypoints.get(name, {})
    if kp.get("occluded", True) or kp.get("x") is None:
        return None
    return (kp["x"], kp["y"])


def _compute_angle(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None:
        return 0.0

    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=float)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]], dtype=float)

    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-6:
        return 0.0

    cos_a = np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def extract_features(person_keypoints):
    ls = _get_xy(person_keypoints, "left_shoulder")
    rs = _get_xy(person_keypoints, "right_shoulder")
    le = _get_xy(person_keypoints, "left_elbow")
    re = _get_xy(person_keypoints, "right_elbow")
    lw = _get_xy(person_keypoints, "left_wrist")
    rw = _get_xy(person_keypoints, "right_wrist")
    nose = _get_xy(person_keypoints, "nose")

    if ls is None and rs is None:
        return None

    if ls and rs:
        center_x = (ls[0] + rs[0]) / 2.0
        center_y = (ls[1] + rs[1]) / 2.0
        shoulder_width = math.dist(ls, rs)
    elif ls:
        center_x, center_y = ls
        shoulder_width = 50.0
    else:
        center_x, center_y = rs
        shoulder_width = 50.0

    shoulder_width = max(shoulder_width, 1e-6)

    def norm(pt):
        if pt is None:
            return (0.0, 0.0)
        return (
            (pt[0] - center_x) / shoulder_width,
            (pt[1] - center_y) / shoulder_width,
        )

    le_n = norm(le)
    re_n = norm(re)
    lw_n = norm(lw)
    rw_n = norm(rw)

    coord_features = [
        le_n[0], le_n[1],
        re_n[0], re_n[1],
        lw_n[0], lw_n[1],
        rw_n[0], rw_n[1],
    ]

    left_elbow_angle = _compute_angle(ls, le, lw) / 180.0
    right_elbow_angle = _compute_angle(rs, re, rw) / 180.0

    lw_raise = norm(lw)[1]
    rw_raise = norm(rw)[1]

    nose_n = norm(nose)
    nose_x = nose_n[0]
    nose_y = nose_n[1]

    angle_features = [
        left_elbow_angle,
        right_elbow_angle,
        lw_raise,
        rw_raise,
        nose_x,
        nose_y,
    ]

    return coord_features + angle_features


if __name__ == "__main__":
    if not os.path.exists(KEYPOINTS_JSON):
        print(f"[ERROR] keypoints_data.json not found.")
        print(f"        Run step1_extract_keypoints.py first.")
        sys.exit(1)

    with open(KEYPOINTS_JSON) as f:
        data = json.load(f)

    print(f"[INFO] Loaded {len(data)} frames from {KEYPOINTS_JSON}")
    tested = 0

    for frame in data:
        for person in frame["persons"]:
            features = extract_features(person["keypoints"])
            if features is not None:
                print(f"\n[TEST] Frame {frame['frame_idx']}, Person {person['person_id']}")
                print(f"       Feature vector ({len(features)} dims):")
                print(f"       {[round(v, 3) for v in features]}")
                tested += 1
            if tested >= 3:
                break
        if tested >= 3:
            break

    if tested == 0:
        print("[WARN] No usable persons found in the first few frames.")
    else:
        print(f"\n[DONE] Feature engineering is working correctly.")
