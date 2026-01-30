from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig

# my code
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.paths import REPO_ROOT, DATASETS_DIR, HF_NAME, POLICIES_DIR

# set up env secrets
import os
from dotenv import load_dotenv
load_dotenv(REPO_ROOT/".env", override=True)

REPO_NAME         = 'so101_pick_pen-bbox'
EXPERIMENT_NAME   = 'bbox_v0'
RESUME_CHECKPOINT = None # none if latest
POLICY_TYPE       = 'act'
DATASET_PATH      = DATASETS_DIR / REPO_NAME
DATASET_ID        = f"{HF_NAME}/{REPO_NAME}"
POLICY_ID         = f"{POLICY_TYPE}-{REPO_NAME}-{EXPERIMENT_NAME}"
POLICY_ROOT       = f"{POLICIES_DIR} / {POLICY_TYPE} - / {REPO_NAME} / {EXPERIMENT_NAME}"

if RESUME_CHECKPOINT is not None:
    POLICY_PATH = POLICIES_DIR / POLICY_TYPE / REPO_NAME / EXPERIMENT_NAME / 'checkpoints' / RESUME_CHECKPOINT / 'pretrained_model'
else:
    POLICY_PATH = POLICIES_DIR / POLICY_TYPE / REPO_NAME / EXPERIMENT_NAME / 'checkpoints' / 'last' / 'pretrained_model'

os.system(f"hf auth login --token {os.getenv('HUGGINGFACE_TOKEN')}")

ds_meta  = LeRobotDatasetMetadata(DATASET_ID, root = DATASET_PATH)

policy_cfg = PreTrainedConfig.from_pretrained(
    pretrained_name_or_path = POLICY_PATH,
    force_download          = False
    )

# train_cfg = TrainPipelineConfig.from_pretrained(
#     pretrained_name_or_path = POLICY_PATH,
#     force_download          = False,
# )

policy = make_policy(policy_cfg, ds_meta)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg      = policy_cfg,
    pretrained_path = policy_cfg.pretrained_path
)

# policy.push_model_to_hub(train_cfg)
policy.push_to_hub(repo_id=POLICY_ID)
preprocessor.push_to_hub(repo_id=POLICY_ID)
postprocessor.push_to_hub(repo_id=POLICY_ID)
