from IPython.display import display, HTML
import pprint
import torch
import numpy as np
import json, statistics, time
from pathlib import Path

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
    

def write_eval_stats(dataset, new_episodes, elapsed_s, num_episodes, filename="eval_results.json"):
    """
    Append new evaluation episodes and update aggregated stats.

    Args:
        dataset: dataset object with .root (pathlib.Path)
        new_episodes: list of dicts with per-episode stats
        elapsed_s: total evaluation time in seconds
        num_episodes: number of episodes in this run
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

    # Append new episodes
    results["per_episode"].extend(new_episodes)

    # Recompute aggregated
    sum_rewards = [ep["sum_reward"] for ep in results["per_episode"] if ep["sum_reward"] is not None]
    max_rewards = [ep["max_reward"] for ep in results["per_episode"] if ep["max_reward"] is not None]
    successes   = [ep["success"] for ep in results["per_episode"] if ep["success"] is not None]

    results["aggregated"] = {
        "avg_sum_reward": statistics.mean(sum_rewards) if sum_rewards else None,
        "avg_max_reward": statistics.mean(max_rewards) if max_rewards else None,
        "pc_success": 100 * statistics.mean(successes) if successes else None,
        "eval_s": elapsed_s,
        "eval_ep_s": elapsed_s / num_episodes if num_episodes else None,
    }


    # Dump
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
