import platform
import getpass
import os
import socket
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower
from src.paths import CALIBS_DIR


# detect system
system = platform.system()   # "Linux", "Windows", "Darwin"
user = getpass.getuser()     # current username
host = socket.gethostname()  # machine hostname

# Linux
# defaults (Linux lab machine)
camera_config = {
    "wrist_cam":OpenCVCameraConfig(
    index_or_path = '/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0',
    fps           = 30,
    width         = 640,
    height        = 480,
    color_mode    = ColorMode.RGB,
    rotation      = Cv2Rotation.NO_ROTATION,
    warmup_s      = 2
    ),
    
    "top_cam": OpenCVCameraConfig(
    index_or_path = '/dev/v4l/by-id/usb-046d_Logitech_BRIO_8F54E371-video-index0',
    fps           = 30,
    width         = 640,
    height        = 480,
    color_mode    = ColorMode.RGB,
    rotation      = Cv2Rotation.NO_ROTATION,
    warmup_s      = 2
    )
}

robot_config = SO101FollowerConfig(
    port            = "/dev/ttyACM0",
    id              = "so_101_follower",
    cameras         = camera_config,
    calibration_dir = CALIBS_DIR
)
teleop_config = SO101LeaderConfig(
    port            = "/dev/ttyACM1",
    id              = "so_101_leader",
    calibration_dir = CALIBS_DIR
)

# override if running on Windows
if system == "Windows":
    
    # setup cv env variables
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0" # TODO = ?
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_DECODING"] = "0"
    os.environ["OPENCV_VIDEOIO_MSMF_FORCE_BGR"] = "1"
    
    camera_config = {
        "wrist_cam": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        "top_cam"  : OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
    }
    robot_config = SO101FollowerConfig(
        port            = "COM7",
        id              = "so_101_follower",
        cameras         = camera_config,
        calibration_dir = CALIBS_DIR
    )
    teleop_config = SO101LeaderConfig(
        port            = "COM8",
        id              = "so_101_leader",
        calibration_dir = CALIBS_DIR
    )

robot = SO101Follower(robot_config)
teleop = SO101Leader(teleop_config)