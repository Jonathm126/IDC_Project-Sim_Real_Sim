from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
import cv2

# paths
from utils.paths import CALIBS_DIR 

camera_config = {
    "wrist_cam": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30)
}

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
    
    # Grab wrist camera image
    frame = observation['wrist_cam']  # numpy array, RGB

    # Convert to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display
    cv2.imshow('Wrist Camera', frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    robot.send_action(action)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

robot.disconnect()
teleop_device.disconnect()
cv2.destroyAllWindows()
