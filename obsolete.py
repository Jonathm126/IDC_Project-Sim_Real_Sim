
# control utils
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import _init_rerun
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.record import record_loop
from lerobot.utils.utils import log_say

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pprint import pprint
from datetime import datetime


robot = SO101Follower(robot_config)
teleop = SO101Leader(teleop_config)
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}
pprint(dataset_features)
dataset_path = DATASETS_DIR / REPO_NAME
if dataset_path.exists():
    dataset=LeRobotDataset(
        repo_id=f"{HF_NAME}/{REPO_NAME}",
        root=f"{DATASETS_DIR}\\{REPO_NAME}"
    )
else:
    dataset = LeRobotDataset.create(
        repo_id=f"{HF_NAME}/{REPO_NAME}",
        root=f"{DATASETS_DIR}\\{REPO_NAME}",
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )
# connect to robots
robot.connect()
teleop.connect()
_, events = init_keyboard_listener()
episode_idx = 0
while episode_idx < NUM_EPISODES and not events["stop_recording"]:
    _init_rerun(session_name="recording", recording_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
)
    log_say(f"Recording episode {episode_idx + 1} of {NUM_EPISODES}")

    record_loop(
        robot=robot,
        events=events,
        fps=FPS,
        teleop=teleop,
        # dataset=dataset,
        dataset=None,
        control_time_s=EPISODE_TIME_SEC,
        single_task=TASK_DESCRIPTION,
        display_data=True,
    )

    # Reset the environment if not stopping or re-recording
    if not events["stop_recording"] and (episode_idx < NUM_EPISODES - 1 or events["rerecord_episode"]):
        log_say("Reset the environment")
        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            teleop=teleop,
            control_time_s=RESET_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=True,
            # dataset=dataset,
            dataset=None
        )

    if events["rerecord_episode"]:
        log_say("Re-recording episode")
        events["rerecord_episode"] = False
        events["exit_early"] = False
        dataset.clear_episode_buffer()
        continue

    dataset.save_episode()
    episode_idx += 1
log_say("Stop recording")
robot.disconnect()
teleop.disconnect()
if PUSH_TO_HUB:
    dataset.push_to_hub()