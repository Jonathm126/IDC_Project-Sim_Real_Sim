
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from src.paths import DATASETS_DIR, HF_NAME, OUTPUTS_DIR

# dataset name
REPO_NAME         = 'so101_car_pick_and_place'

ds = LeRobotDataset(
    repo_id = f"{HF_NAME}/{REPO_NAME}",
    root = DATASETS_DIR / REPO_NAME,
    download_videos = False
)

indices = ds.meta.episodes['dataset_from_index']

images = []
for idx in indices:
    images.append(ds[idx]['observation.images.top_cam'])

stack_img = torch.stack(images)
avg_img = torch.mean(stack_img, dim = 0)
avg_img_np = avg_img.permute(1,2,0).numpy()
avg_img_np = (avg_img_np * 255).astype(np.uint8)
avg_img_pil = Image.fromarray(avg_img_np)

save_path = f"{OUTPUTS_DIR}/avg_img/{REPO_NAME}.jpg" 
avg_img_pil.save(save_path)
print(f"Average image saved to {save_path}")
