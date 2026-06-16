MODEL_PATH = "models/best_ckpt_fp16.pth"

CONF_THRESH = 0.10
IOU_THRESH = 0.45
MAX_DET = 50

INPUT_SIZE = (640, 640)

NUM_CLASSES = 13

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