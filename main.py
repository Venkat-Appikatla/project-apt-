from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import torch
import json
import os
import gdown

from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.data_augment import ValTransform

app = FastAPI(title="Food Detection API")

# ---------------- CONFIG ----------------

MODEL_PATH = "models/best_ckpt_fp16.pth"
EXP_PATH   = "food_yolox_l.py"

CONF_THRESH  = 0.25        # raised from 0.05 → reduces false positives
IOU_THRESH   = 0.45
INPUT_SIZE   = (640, 640)
NUM_CLASSES  = 13
MAX_DET      = 50
USE_FP16     = True          # matches your fp16 checkpoint

CLASS_NAMES = [
    "idly",
    "dosa",
    "rice",
    "boiled_eggs",
    "chuteny",
    "dal",
    "curd",
    "chapathi",
    "kakarakaya fry",
    "egg curry",
    "pulihora",
    "sambar",
    "unknown"
]

FILTER_CLASSES = {"unknown"}   # classes to suppress from results

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODEL ----------------

exp = get_exp(EXP_PATH, None)

exp.num_classes = NUM_CLASSES
exp.test_conf   = CONF_THRESH
exp.nmsthre     = IOU_THRESH
exp.test_size   = INPUT_SIZE

model = exp.get_model()

# Download model from Google Drive if missing
if not os.path.exists(MODEL_PATH):
    os.makedirs("models", exist_ok=True)

    gdown.download(
    id="1dNDsB94ooN95E2r-Z7gT5O0VpDKB4-8F",
    output=MODEL_PATH,
    quiet=False
)

# Verify model exists
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model download failed")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

state_dict = ckpt["model"] if "model" in ckpt else ckpt

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

# Enable FP16 if checkpoint was trained in half precision
if USE_FP16 and DEVICE == "cuda":
    model.half()

preproc = ValTransform(legacy=False)

with open("nutrition.json", "r", encoding="utf-8") as f:
    NUTRITION_DB = json.load(f)

print("[INFO] Nutrition database loaded")

print(f"[INFO] Model loaded on {DEVICE} | FP16={USE_FP16 and DEVICE == 'cuda'}")


# ---------------- HELPERS ----------------

def preprocess(img: np.ndarray) -> torch.Tensor:
    # Use original OpenCV BGR image exactly like YOLOX demo

    img_t, _ = preproc(img, None, INPUT_SIZE)

    tensor = torch.from_numpy(img_t).unsqueeze(0).float().to(DEVICE)

    if USE_FP16 and DEVICE == "cuda":
        tensor = tensor.half()

    return tensor


def run_inference(img: np.ndarray) -> list:
    h, w = img.shape[:2]
    tensor = preprocess(img)

    with torch.no_grad():
        outputs = model(tensor)
        outputs = postprocess(
            outputs,
            num_classes=NUM_CLASSES,
            conf_thre=CONF_THRESH,
            nms_thre=IOU_THRESH
        )

    if outputs[0] is None:
        return []

    # Correct scale: how much the image was resized to fit INPUT_SIZE
    scale = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)

    detections = []

    for det in outputs[0].cpu().float().numpy():
        x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det

        cls_name   = CLASS_NAMES[int(cls_id)]
        confidence = round(float(obj_conf * cls_conf), 4)

        # Skip suppressed classes
        if cls_name in FILTER_CLASSES:
            continue

        # Scale bbox back to original image coords and clamp
        bx1 = round(max(0.0,  float(x1) / scale), 2)
        by1 = round(max(0.0,  float(y1) / scale), 2)
        bx2 = round(min(float(w), float(x2) / scale), 2)
        by2 = round(min(float(h), float(y2) / scale), 2)

        nutrition = NUTRITION_DB.get(cls_name, {})
        detections.append({
            "class":      cls_name,
            "confidence": confidence,
            "bbox":       [bx1, by1, bx2, by2],
            "nutrition": nutrition
        })

    # Sort by confidence descending and cap at MAX_DET
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)[:MAX_DET]

    return detections


# ---------------- ROUTES ----------------

@app.get("/")
def home():
    return {"message": "Food API Running"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": DEVICE,
        "fp16":   USE_FP16 and DEVICE == "cuda",
        "conf_threshold": CONF_THRESH
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image — could not decode file"}

        detections = run_inference(img)

        return {
            "filename":         file.filename,
            "image_shape":      list(img.shape),
            "total_detections": len(detections),
            "detections":       detections
        }

    except Exception as e:
        return {"error": str(e)}
