from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from lerobot.cameras.configs import ColorMode, Cv2Rotation
import cv2

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# paths
from utils.paths import CALIBS_DIR 

camera_config = {
    "wrist_cam": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    "top_cam": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30, rotation=Cv2Rotation.NO_ROTATION),
}

cam = True

robot_config = SO101FollowerConfig(
    port="COM7",
    id="so_101_follower",
    cameras = camera_config,
    calibration_dir = CALIBS_DIR
)

teleop_config = SO101LeaderConfig(
    port="COM8",
    id="so_101_leader",
    calibration_dir = CALIBS_DIR
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)

robot.connect()
teleop_device.connect()

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

        # Display
        cv2.imshow(" | ".join(camera_config.keys()), combined)
    
    robot.send_action(action)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

robot.disconnect()
teleop_device.disconnect()
cv2.destroyAllWindows()
