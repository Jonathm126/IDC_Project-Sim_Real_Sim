import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.paths import REPO_ROOT, DATASETS_DIR, HF_NAME
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dotenv import load_dotenv
load_dotenv(REPO_ROOT/".env", override=True)

os.system(f"hf auth login --token {os.getenv('HUGGINGFACE_TOKEN')}")

REPO_NAME = 'so101_car_pick_and_place'

dataset_path = f"{DATASETS_DIR}/{REPO_NAME}"
repo_id      = f"{HF_NAME}/{REPO_NAME}"

ds = LeRobotDataset(repo_id = repo_id, root = dataset_path)
ds.pull_from_repo()
