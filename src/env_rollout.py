import gymnasium as gym
import torch
import numpy as np

# lerobot
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from lerobot.envs.configs import EnvConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.teleoperators import Teleoperator
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# utils
import time
from src.utils import process_obs_to_np
# from src.rr_logger import RRLogger TODO switch to my rr logger

def env_rollout(
    # manage all rollouts with logic
    display_rerun: bool,
    env_cfg      : EnvConfig,
    env          : gym.Env,
    num_episodes : int,
    policy       : PreTrainedPolicy | None = None,
    teleop       : Teleoperator | None = None,
    dataset      : LeRobotDataset | None = None
):
    '''
    Rollouts in env with inputs from 
    display_rerun(bool): Use rerun for live display
    env_cfg(EnvConfig): config file for the env
    env(gym.Env): env
    num_episodes: int
    policy(PreTrainedPolicy): optional
    teleop(Teleoperator): optional
    Note - must have either policy or teleop to get actions, but not both
    '''
    assert (policy is not None) ^ (teleop is not None), "Must provide either policy or teleop, but not both."
    
    # inits
    listener, events = init_keyboard_listener()
    frame_time = 1.0 / env_cfg.fps
    stats_all = []
    
    # init dataset
    pass
    
    try:
        for ep in range(num_episodes):
            stats = one_rollout(
                display_rerun = display_rerun,
                events        = events,
                env           = env,
                frame_time    = frame_time,
                policy        = policy,
                teleop        = teleop,
                dataset       = dataset,
            )
            stats_all.append(stats)
            print(f"Episode {ep+1}/{num_episodes}: reward={stats['reward']:.2f}, success={stats['success']}")

            if events["stop_recording"]:
                break
    finally:
            if listener:
                listener.stop()
    
    rewards = [s["reward"] for s in stats_all]
    successes = sum(s["success"] for s in stats_all)
    return {
        "episodes": stats_all,
        "avg_reward": np.mean(rewards) if rewards else 0.0,
        "success_rate": successes / len(stats_all) if stats_all else 0.0,
    }


def one_rollout(
    display_rerun: bool,
    events    : dict,
    env       : gym.Env,
    frame_time: float,
    policy    : PreTrainedPolicy | None = None,
    teleop    : Teleoperator | None = None,
    dataset   : LeRobotDataset | None = None
) -> dict:
    # init rr
    if display_rerun:
        _init_rerun(session_name="teleop_env", recording_id=time.time())
    
    # reset policy
    if policy is not None:
        policy.reset()
        policy.eval()
    
    # reset env
    obs, _ = env.reset()
    episode_reward = 0.0
    
    while True:
        loop_start = time.time()
        
        # Check keyboard flags
        if events["exit_early"]:
            events["exit_early"] = False
            break
        if events["rerecord_episode"]:
            pass
        if events["stop_recording"]:
            events["stop_recording"] = False
            break
        
        # 1. Get action, map to env space
        if policy is not None:
            with torch.inference_mode():
                action = policy.select_action(obs)
        else:
            action = teleop.get_action()
            # cast to torch
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(list(action.values()), dtype=torch.float32).cpu()
        
        # 2. Apply to env
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        
        # 3. Log to rerun
        if display_rerun:
            # process observation
            obs_to_log = process_obs_to_np(obs)
            
            # log metrics
            obs_to_log["reward"] = reward
            obs_to_log["success"] = terminated
            obs_to_log["done"]   = done
            obs_to_log["render"] = env.render()
            action_to_log        = {"action": np.array(action.squeeze().cpu())}
            log_rerun_data(obs_to_log, action_to_log)
        
        # 4. log to dataset
        pass
        
        # 5. Stopping logic
        if done:
            break 
        
        # 6. Maintain FPS
        if teleop is not None:
            elapsed = time.time() - loop_start
            time.sleep(max(0, frame_time - elapsed))
    
    return {
    "reward": episode_reward,
    "done": done,
    "success": terminated,
    }