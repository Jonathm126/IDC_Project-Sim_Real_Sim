import random
import torchvision.transforms.functional as F
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

# paths
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from paths import DATASETS_DIR, HF_NAME


if __name__ == "__main__":
    # params
    REPO_NAME   = 'so101_car_pick_and_place'
    CAM_NAME    = "observation.images.top_cam"
    num_samples = 200                               # random frames to sample
    # target_size = (640, 480)                        # resize
    target_size = None
    
    # paths
    repo_id    = f"{HF_NAME}/{REPO_NAME}"
    ds_root    = f"{DATASETS_DIR}/{REPO_NAME}"
    output_dir = Path(f"{ds_root}/bboxes/samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # sample
    dataset = LeRobotDataset(f"{HF_NAME}/{REPO_NAME}", root=f"{DATASETS_DIR}/{REPO_NAME}")
    print(f"Sampling {num_samples} frames from {len(dataset)}")
    sample_indices = sorted(random.sample(range(len(dataset)), num_samples))

    for i, idx in enumerate(sample_indices):
        sample = dataset[idx]
        img_tensor = sample[CAM_NAME] 

        # Convert tensor → PIL Image
        img_pil = F.to_pil_image(img_tensor)

        # Resize if requested
        if target_size is not None:
            img_pil = img_pil.resize(target_size)

        img_pil.save(output_dir / f"frame_{i:06d}.png")

    print(f"Done. Saved {num_samples} frames to {output_dir}")
