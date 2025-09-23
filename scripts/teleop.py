import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from pathlib import Path
import cv2
import time

# paths
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.paths import CALIBS_DIR 

# Camera configs
camera_config = {
    "wrist_cam": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    "top_cam": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30, rotation=Cv2Rotation.NO_ROTATION),
}

cam = True

# Robot + teleop configs
robot_config = SO101FollowerConfig(
    port="COM7",
    id="so_101_follower",
    cameras=camera_config,
    calibration_dir=CALIBS_DIR
)

teleop_config = SO101LeaderConfig(
    port="COM8",
    id="so_101_leader",
    calibration_dir=CALIBS_DIR
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)

robot.connect()
teleop_device.connect()

# FPS tracking
frame_count = 0
start_time = time.time()
fps = 0.0

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    
    if cam:
        frames_bgr = []
        for cam_name in camera_config.keys():
            frame_rgb = observation[cam_name]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frames_bgr.append(frame_bgr)
        
        # Resize frames to same height
        target_height = min(f.shape[0] for f in frames_bgr)
        frames_bgr = [
            cv2.resize(f, (int(f.shape[1] * target_height / f.shape[0]), target_height))
            for f in frames_bgr
        ]
        
        combined = cv2.hconcat(frames_bgr)  

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # Overlay FPS on combined frame
        cv2.putText(
            combined,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Display
        cv2.imshow(" | ".join(camera_config.keys()), combined)
    
    robot.send_action(action)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

robot.disconnect()
teleop_device.disconnect()
cv2.destroyAllWindows()
