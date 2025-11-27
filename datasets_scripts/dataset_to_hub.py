from lerobot.datasets.lerobot_dataset import LeRobotDataset

# my code
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.paths import REPO_ROOT, DATASETS_DIR, HF_NAME, EVAL_DIR

# set up env secrets
import os
from dotenv import load_dotenv
load_dotenv(REPO_ROOT/".env", override=True)

REPO_NAME    = 'so101_car_pick_and_place'
DATASET_PATH = DATASETS_DIR/REPO_NAME
DATASET_ID   = f"{HF_NAME}/{REPO_NAME}"

os.system(f"hf auth login --token {os.getenv('HUGGINGFACE_TOKEN')}")
# ds = LeRobotDataset(repo_id = f"{HF_NAME}/{REPO_NAME}", root=f"{DATASETS_DIR}/{REPO_NAME}")

ds = LeRobotDataset(repo_id=f"{HF_NAME}/eval_so101_car_pick_and_place-bbox-yolo_v0", root = EVAL_DIR/'act'/'so101_car_pick_and_place-bbox'/'yolo_v0-real_v0')
ds.push_to_hub()