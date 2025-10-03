from pathlib import Path
import cv2
import time

# paths
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from robot.robot_config import robot, teleop

robot.connect()
teleop.connect()

# FPS tracking
frame_count = 0
start_time = time.time()
fps = 0.0

while True: 
    observation = robot.get_observation()
    action = teleop.get_action()
    
    if robot.cameras is not None:
        frames_bgr = []
        for cam_name in robot.cameras.keys():
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
        cv2.imshow(" | ".join(robot.cameras.keys()), combined)
    
    robot.send_action(action)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

robot.disconnect()
teleop.disconnect()
cv2.destroyAllWindows()
