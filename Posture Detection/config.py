import os

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_VIDEO_DIR   = os.path.join(DATA_DIR, "raw_videos")
KEYPOINTS_DIR   = os.path.join(DATA_DIR, "keypoints")
DATASET_DIR     = os.path.join(DATA_DIR, "datasets")

MODELS_DIR      = os.path.join(BASE_DIR, "models")
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")

INPUT_VIDEO         = os.path.join(RAW_VIDEO_DIR,  "classroom_video.mp4")
KEYPOINTS_JSON      = os.path.join(KEYPOINTS_DIR,  "keypoints_data.json")
LABELS_JSON         = os.path.join(KEYPOINTS_DIR,  "labels.json")
X_TRAIN_FILE        = os.path.join(DATASET_DIR,    "X_train.npy")
Y_TRAIN_FILE        = os.path.join(DATASET_DIR,    "y_train.npy")
CLASSIFIER_MODEL    = os.path.join(MODELS_DIR,     "posture_classifier.pth")
OUTPUT_VIDEO        = os.path.join(OUTPUT_DIR,     "output_annotated.mp4")

YOLO_MODEL = "yolov8n-pose.pt"

KEYPOINT_NAMES = {
    0:  "nose",
    5:  "left_shoulder",
    6:  "right_shoulder",
    7:  "left_elbow",
    8:  "right_elbow",
    9:  "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
}

LABEL_MAP = {
    "attentive":    0,
    "distracted":   1,
    "hand_raising": 2,
    "writing":      3,
}

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

LABEL_COLORS = {

    0: (0,   255,   0),   
    1: (0,     0, 255),   
    2: (0,   165, 255),   
    3: (0,   255, 255),   
}

NUM_CLASSES    = len(LABEL_MAP)  
FEATURE_SIZE   = 14                 

KEYPOINT_CONF_THRESHOLD    = 0.5   
CLASSIFIER_CONF_THRESHOLD  = 0.7   

EXTRACT_EVERY_N_FRAMES     = 3     
INFERENCE_EVERY_N_FRAMES   = 2     

EPOCHS         = 50
BATCH_SIZE     = 32
LEARNING_RATE  = 0.001
TRAIN_SPLIT    = 0.8               
DROPOUT_RATE   = 0.4

SMOOTH_WINDOW  = 10                

OUTPUT_FPS     = 15
