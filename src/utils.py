from IPython.display import display, HTML
import pprint
import torch
import numpy as np
import json
import statistics
from pathlib import Path
import shutil

def scroll_print(data): 
    display(HTML(f"""<div style="max-height:300px; overflow:auto; border:1px solid #ccc; padding:10px;">
    <pre>{pprint.pformat(data, indent=2, width=80)}</pre>
    </div>"""))

def process_obs_to_np(obs):
    return {
    k: (
        (
            v.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            if v.dtype == torch.uint8
            else (v.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        )
        if "images" in k
        else v.detach().cpu().squeeze(0).float().numpy()
    )
    for k, v in obs.items()
}
    

def write_eval_stats(dataset, episodes, elapsed_s = None, filename="eval_results.json"):
    """
    Append evaluation results and update aggregated stats.

    Args:
        dataset: dataset object with .root (pathlib.Path)
        episodes: list of dicts with episode stats (partial dicts allowed)
        elapsed_s: total evaluation time in seconds
        filename: json filename (default: eval_results.json)
    """
    meta_dir = Path(dataset.root) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    json_path = meta_dir / filename

    # Load old results if exist
    if json_path.exists():
        with open(json_path, "r") as f:
            results = json.load(f)
    else:
        results = {"per_episode": []}

    # Normalize new episodes: fill missing keys with None
    default_keys = ["episode_idx", "sum_reward", "max_reward", "success", "seed"]
    normalized_eps = []
    for ep in episodes:
        full_ep = {k: ep.get(k, None) for k in default_keys}
        normalized_eps.append(full_ep)

    # Append
    results["per_episode"].extend(normalized_eps)

    # Recompute aggregated stats
    sum_rewards = [ep["sum_reward"] for ep in results["per_episode"] if ep["sum_reward"] is not None]
    max_rewards = [ep["max_reward"] for ep in results["per_episode"] if ep["max_reward"] is not None]
    successes   = [ep["success"] for ep in results["per_episode"] if ep["success"] is not None]
    
    num_episodes = len(results["per_episode"])

    results["aggregated"] = {
        "avg_sum_reward": statistics.mean(sum_rewards) if sum_rewards else None,
        "avg_max_reward": statistics.mean(max_rewards) if max_rewards else None,
        "pc_success"    : 100 * statistics.mean(successes) if successes else None,
        "eval_s"        : elapsed_s if elapsed_s else None,
        "eval_ep_s": (elapsed_s / num_episodes) if (elapsed_s is not None and num_episodes != 0) else None,
    }

    # Save
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)



def check_resume(dataset_path: Path) -> bool:
    """
    Check if training should resume from dataset_path.
    If info.json exists and total_episodes == 0 → delete dataset_path and return False.
    Otherwise → return True if dataset_path exists, else False.
    """
    if not dataset_path.exists():
        return False

    info_file = dataset_path / "meta" / "info.json"
    if info_file.exists():
        with open(info_file, "r") as f:
            meta = json.load(f)
        if meta.get("total_episodes", 0) == 0:
            shutil.rmtree(dataset_path)
            return False

    return True

