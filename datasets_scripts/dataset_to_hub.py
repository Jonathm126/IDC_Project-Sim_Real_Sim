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

REPO_NAME    = 'so101_pick_pen'
DATASET_PATH = DATASETS_DIR/REPO_NAME
EXPERIMENT_ID = 'baseline_v0'
EVAL_ID = 'real_v0'
DATASET_ID   = f"{HF_NAME}/{REPO_NAME}"

os.system(f"hf auth login --token {os.getenv('HUGGINGFACE_TOKEN')}")
ds = LeRobotDataset(repo_id = f"{HF_NAME}/{REPO_NAME}", root=f"{DATASETS_DIR}/{REPO_NAME}")
# ds = LeRobotDataset(repo_id = f"{HF_NAME}/eval_{REPO_NAME}-{EXPERIMENT_ID}-{EVAL_ID}", root = EVAL_DIR/'act'/REPO_NAME/f'{EXPERIMENT_ID}-{EVAL_ID}')
ds.push_to_hub()