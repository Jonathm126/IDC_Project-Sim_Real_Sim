from lerobot.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

from datetime import datetime

from utils.paths import CALIBS_DIR 

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
CALIBRATION_BASE = CALIBS_DIR / timestamp
CALIBRATION_BASE.mkdir(parents=True, exist_ok=True)

robot_config = SO101FollowerConfig(
    port="COM7",
    id="so_101_follower",
    calibration_dir=str(CALIBRATION_BASE)  
)

teleop_config = SO101LeaderConfig(
    port="COM8",
    id="so_101_leader",
    calibration_dir=str(CALIBRATION_BASE)  
)

follower = SO101Follower(robot_config)
leader = SO101Leader(teleop_config)

follower.connect(calibrate=False)
follower.calibrate()
follower.disconnect()

# leader.connect(calibrate=False)
# leader.calibrate()
# leader.disconnect()