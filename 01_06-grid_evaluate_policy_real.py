# policy
from lerobot.configs.policies import PreTrainedConfig

# record utils
from lerobot.record import record, RecordConfig, DatasetRecordConfig

# torch
from torch import cuda

# utils
import pprint
import os
import json
import random
from dotenv import load_dotenv
from itertools import product


# my code
from robot.robot_config import robot_config
from robot.robot_const import TABLE_START_POSE_OPEN, FOLDED_START_POSE
from src.paths import REPO_ROOT, HF_NAME, POLICIES_DIR, EVAL_DIR
from src.utils import check_resume

# set up env secrets
load_dotenv(REPO_ROOT / ".env", override=True)

# cuda
device = "cuda" if cuda.is_available() else "cpu"
print(f"Using device: {device}")


# --- GLOBAL PARAMS ---
PUSH_TO_HUB      = False
SAVE_TO_DATASET  = True
REPO_NAME        = 'so101_car_pick_and_place'
EXPERIMENT_NAME  = '50_episodes_v2'
POLICY_TYPE      = 'act'
FPS              = 30
EPISODE_TIME_SEC = 60
RESET_TIME_SEC   = 5
EVAL_TYPE        = '12_runs_grid'

# --- CONFIGURABLE PARAM SWEEP ---
n_action_steps_list = [15, 30, 60, 100]
# checkpoints = ["40000", "60000"]
POLICY_CHECKPOINT = 60000 # 60000
temporal_ensemble_coeff = None
# temporal_ensemble_coeff_list  = [False, True]

# starting positions
bin_positions    = [1, 2, 3]
start_positions = [1, 2, 3, 4, 5]
rotations        = [0, 90]
invalid_pairs = {
    (1, 2),
    (2, 3),
    (3, 4)
}
# Filter out invalid (bin, start) pairs
all_cases = product(bin_positions, start_positions, rotations)
valid_test_cases = [
    (bin_pos, start_pos, rot)
    for (bin_pos, start_pos, rot) in all_cases
    if (bin_pos, start_pos) not in invalid_pairs
]

if PUSH_TO_HUB:
    os.system(f"huggingface-cli login --token {os.getenv('HUGGINGFACE_TOKEN')}")

# resolve path or HF id
if POLICY_CHECKPOINT is None:
    raise ValueError('specify exact checkpoint')
policy_path = POLICIES_DIR / POLICY_TYPE / REPO_NAME / EXPERIMENT_NAME / "checkpoints" / POLICY_CHECKPOINT / "pretrained_model"
if not policy_path.exists():
    raise ValueError('Policy path does not exist')

# build policy
policy_config = PreTrainedConfig.from_pretrained(policy_path)
policy_config.pretrained_path = policy_path 

# build dataset
dataset_path = EVAL_DIR / POLICY_TYPE / REPO_NAME / f"{EXPERIMENT_NAME}-{EVAL_TYPE}"
resume = check_resume(dataset_path)

dscfg = DatasetRecordConfig(
    repo_id                             = f"{HF_NAME}/eval_{REPO_NAME}_{EXPERIMENT_NAME}_{EVAL_TYPE}",
    single_task                         = f"eval dataset for {REPO_NAME} with policy {EXPERIMENT_NAME}, mode = {EVAL_TYPE}",
    root                                = dataset_path.__str__(),
    fps                                 = FPS,
    episode_time_s                      = EPISODE_TIME_SEC,
    reset_time_s                        = RESET_TIME_SEC,
    num_episodes                        = len(valid_test_cases),
    video                               = True,
    push_to_hub                         = PUSH_TO_HUB,
    private                             = True,
    num_image_writer_processes          = 0,
    num_image_writer_threads_per_camera = 4,
    video_encoding_batch_size           = 1,
)

rc = RecordConfig(
    robot        = robot_config,
    dataset      = dscfg,
    teleop       = None,
    policy       = policy_config,
    display_data = True,
    play_sounds  = True,
    resume       = resume
)

# start iterating over chinks
for n_action_steps in n_action_steps_list:
    # policy overrides
    policy_config.n_action_steps = n_action_steps
    policy_config.temporal_ensemble_coeff = temporal_ensemble_coeff
    
    # run the record loop
    record(rc, 
    save_to_ds    = SAVE_TO_DATASET,
    reset_pose    = FOLDED_START_POSE,
    give_feedback = True,
    log_to_file   = False)
    
    # after running 12 TC's - add 