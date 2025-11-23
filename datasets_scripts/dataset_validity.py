# dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# utils
import torch

# my code
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.paths import DATASETS_DIR, HF_NAME

### FILL HERE ###
REPO_NAME       = 'so101_car_pick_and_place'
EXPERIMENT_NAME = '50_episodes_v0'
DATASET_PATH    = DATASETS_DIR/REPO_NAME
DATASET_ID      = f"{HF_NAME}/{REPO_NAME}"

def validate_dataset(ds, batch_size=1, num_workers=0):
    from torch.utils.data import DataLoader

    test_dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    it = iter(test_dl)

    errors = []
    for idx in range(len(ds)):
        try:
            _ = next(it)
        except Exception as e:
            errors.append((idx, repr(e)))

    if errors:
        print(f"❌ Found {len(errors)} invalid samples")
        for idx, err in errors:
            print(f"  - index {idx}: {err}")
    else:
        print("✅ Dataset fully valid")

    return errors

def map_bad_indices_to_episodes(bad_indices, ds_meta):
    # Build start/end index ranges for each episode
    starts, ends = {}, {}
    curr = 0
    for ep_id in ds_meta.episodes:   # keys, e.g. 0, 1, 2...
        L = ds_meta.episodes[ep_id]["length"]
        starts[ep_id] = curr
        ends[ep_id] = curr + L - 1
        curr += L

    # Map bad indices to episodes
    bad_episodes = {}
    for idx in bad_indices:
        for ep_id in ds_meta.episodes:
            if starts[ep_id] <= idx <= ends[ep_id]:
                if ep_id not in bad_episodes:
                    bad_episodes[ep_id] = []
                # store frame index relative to episode
                bad_episodes[ep_id].append(idx - starts[ep_id])
                break

    return bad_episodes

# select episodes
ds_meta= LeRobotDatasetMetadata(DATASET_ID, root = DATASET_PATH)
bad_eps = {}
episodes = [ep for ep in range(len(ds_meta.episodes)) if ep not in bad_eps]
ds = LeRobotDataset(repo_id = f"{HF_NAME}/{REPO_NAME}", root=f"{DATASETS_DIR}/{REPO_NAME}") # episodes=episodes,

errors = validate_dataset(ds, batch_size=1, num_workers=0)

if errors:
    # mitigation: drop bad samples
    bad_indices = [i for i, _ in errors]
    print("Bad indices:", bad_indices)
    # Option 1: filter dataset
    ds = torch.utils.data.Subset(ds, [i for i in range(len(ds)) if i not in bad_indices])
else:
    print("Dataset ready for training")

if errors:
    bad_episodes = map_bad_indices_to_episodes(bad_indices, ds_meta)
    for ep_id, frames in bad_episodes.items():
        print(f"Episode {ep_id} bad frames: {min(frames)}–{max(frames)} "
            f"(total {len(frames)} frames)")