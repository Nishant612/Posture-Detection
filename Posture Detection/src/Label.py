import cv2
import json
import os
import sys
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INPUT_VIDEO, LABELS_JSON, KEYPOINTS_DIR

KEY_TO_LABEL = {
    ord('a'): "attentive",
    ord('d'): "distracted",
    ord('h'): "hand_raising",
    ord('w'): "writing",
}

COLORS = {
    "attentive":    (0,   255,   0),
    "distracted":   (0,     0, 255),
    "hand_raising": (0,   165, 255),
    "writing":      (0,   255, 255),
}


def save_labels(segments, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(segments, f, indent=2)


def label_video():

    if not os.path.exists(INPUT_VIDEO):
        print(f"\n[ERROR] Video not found: {INPUT_VIDEO}")
        print("        Put your video there and name it classroom_video.mp4\n")
        sys.exit(1)

    os.makedirs(KEYPOINTS_DIR, exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"\n[ERROR] Could not open video: {INPUT_VIDEO}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0



    segments  = []
    active    = {}
    frame_idx = 0
    paused    = False
    frame     = None

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] End of video reached.")
                break
            frame_idx += 1

        if frame is None:
            continue

        display = frame.copy()

        y = 40
        if active:
            for label, start_f in active.items():
                color = COLORS.get(label, (255, 255, 255))
                cv2.rectangle(display, (0, y - 22), (520, y + 6), (20, 20, 20), -1)
                cv2.putText(display,
                            f"● RECORDING: {label.upper()}  (frame {start_f})",
                            (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                y += 35
        else:
            cv2.putText(display,
                        "Press A / D / H / W to start a segment",
                        (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        cv2.putText(display,
                    f"Segments saved: {len(segments)}   Frame: {frame_idx}/{total}",
                    (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if paused:
            cv2.putText(display, "PAUSED",
                        (display.shape[1] - 120, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Step 3 - Labelling  |  Q=quit  S=save  SPACE=pause",
                   display)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord(' '):
            paused = not paused

        elif key == ord('s'):
            save_labels(segments, LABELS_JSON)

        elif key in KEY_TO_LABEL:
            label = KEY_TO_LABEL[key]

            if label in active:
                seg = {
                    "frame_start": int(active[label]),
                    "frame_end":   int(frame_idx),
                    "label":       label,
                }
                segments.append(seg)
                del active[label]
                duration = (seg["frame_end"] - seg["frame_start"]) / fps
                save_labels(segments, LABELS_JSON)
            else:
                active[label] = frame_idx


    cap.release()
    cv2.destroyAllWindows()

    if active:
        for label, start_f in active.items():
            seg = {
                "frame_start": int(start_f),
                "frame_end":   int(frame_idx),
                "label":       label,
            }
            segments.append(seg)
            print(f"  [AUTO-CLOSED] {label}  frames {start_f} → {frame_idx}")

    if len(segments) == 0:
        print("\n[WARN] No segments were created!")
        print("       Creating SAMPLE labels.json so pipeline does not break.")
        print("       Replace with real labels later for good accuracy.\n")
        segments = [
            {"frame_start": 0,   "frame_end": 150, "label": "attentive"},
            {"frame_start": 150, "frame_end": 300, "label": "distracted"},
            {"frame_start": 300, "frame_end": 450, "label": "hand_raising"},
            {"frame_start": 450, "frame_end": 600, "label": "writing"},
        ]

    save_labels(segments, LABELS_JSON)

    counts = Counter(s["label"] for s in segments)
    for lbl, cnt in counts.items():
        secs = sum(s["frame_end"] - s["frame_start"]
                   for s in segments if s["label"] == lbl) / fps



if __name__ == "__main__":
    label_video()