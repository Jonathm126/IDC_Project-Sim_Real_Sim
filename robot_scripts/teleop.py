from pathlib import Path
import cv2
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

# paths
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from robot.robot_config import robot, robot_ext, teleop

def draw_plot(data_dict, width=600, height=400, margin=50, y_ticks=5):
    """
    Draw rolling plots with Y-axis scale indicators.

    data_dict: dict[param_name] = list of values (all same length)
    width, height: size of the image
    margin: space for axis labels
    y_ticks: number of ticks on Y-axis
    """

    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # white bg

    # Draw axes
    cv2.line(img, (margin, height - margin), (width - margin, height - margin), (0,0,0), 2)  # X axis
    cv2.line(img, (margin, margin), (margin, height - margin), (0,0,0), 2)  # Y axis

    max_len = max(len(vals) for vals in data_dict.values())
    max_val = max(max(vals) for vals in data_dict.values())
    min_val = min(min(vals) for vals in data_dict.values())
    val_range = max_val - min_val if max_val != min_val else 1

    # Draw Y-axis ticks and labels
    for i in range(y_ticks + 1):
        y = height - margin - int(i * (height - 2*margin) / y_ticks)
        val = min_val + i * val_range / y_ticks
        cv2.line(img, (margin - 5, y), (margin, y), (0,0,0), 2)
        cv2.putText(img, f"{val:.1f}", (5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    colors = {
        "Present_Current": (255, 0, 0),
        "Moving_Velocity": (0, 255, 0),
        "Present_Velocity": (0, 0, 255),
        "Present_Load": (255, 128, 0),
        "FPS": (0, 128, 255)
    }

    # Draw lines for each param
    for param, vals in data_dict.items():
        if len(vals) < 2:
            continue

        points = []
        for i, v in enumerate(vals):
            x = margin + int(i * (width - 2*margin) / max_len)
            y = height - margin - int((v - min_val) * (height - 2*margin) / val_range)
            points.append((x, y))

        cv2.polylines(img, [np.array(points, dtype=np.int32)], False, colors.get(param, (0,0,0)), 2)

        # Put param name with color (top right corner)
        cv2.putText(img, param, (width - margin - 100, margin + 20 * list(data_dict.keys()).index(param)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.get(param, (0,0,0)), 2)

    return img

# === Torque and FPS readout flag ===
ENABLE_TORQUE_READOUT = True  
MAX_HISTORY = 100

# Data buffer for FPS
fps_data = deque(maxlen=MAX_HISTORY)

# motor and param info
param_names = ["current"] # , "Moving_Velocity", "Present_Velocity", "Present_Load"s
motor_names = list(robot.bus.motors.keys())

if ENABLE_TORQUE_READOUT:
    robot = robot_ext
    
    motor_data = {
        motor: {param: deque(maxlen=MAX_HISTORY) for param in param_names}
        for motor in motor_names
    }


# --- Setup Matplotlib Subplots ---
# num_rows = 1 + len(motor_names)  # One for FPS + one per motor
# fig, axes = plt.subplots(num_rows, 1, figsize=(12, 3 * num_rows), sharex=True)
# if num_rows == 1:
#     axes = [axes]  # Ensure it's iterable if only FPS row

# plt.ion()
# plt.tight_layout()

# -- connect to robot ---
robot.connect()
teleop.connect()

prev_time = time.time()

while True:
    # --- Calculate FPS per frame ---
    current_time = time.time()
    delta_time = current_time - prev_time
    prev_time = current_time
    fps = 1.0 / delta_time if delta_time > 0 else 0.0
    fps_data.append(fps)

    # --- Read robot state ---
    observation = robot.get_observation()
    action = teleop.get_action()

    # --- Show camera(s) ---
    if robot.cameras is not None:
        frames_bgr = []
        for cam_name in robot.cameras.keys():
            frame_rgb = observation[cam_name]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            frames_bgr.append(frame_bgr)

        # Resize to match heights
        target_height = min(f.shape[0] for f in frames_bgr)
        frames_bgr = [
            cv2.resize(f, (int(f.shape[1] * target_height / f.shape[0]), target_height))
            for f in frames_bgr
        ]
        combined = cv2.hconcat(frames_bgr)
        cv2.imshow(" | ".join(robot.cameras.keys()), combined)

    # --- Plotting ---
    # # Plot FPS (always at axes[0])
    # axes[0].clear()
    # axes[0].plot(fps_data, label="FPS", color='blue')
    # axes[0].set_title("FPS")
    # axes[0].legend()
    # if fps_data:
    #     y_min = min(fps_data)
    #     y_max = max(fps_data)
    #     margin = 0.1 * (y_max - y_min) if y_max != y_min else 1
    #     axes[0].set_ylim(y_min - margin, y_max + margin)
    # axes[0].grid(True)

    # # Plot motor parameters (if enabled)
    # if ENABLE_TORQUE_READOUT:
    #     for i, motor in enumerate(motor_names):
    #         ax = axes[i + 1]  # +1 to skip FPS row
    #         ax.clear()
    #         for param in param_names:
    #             ax.plot(motor_data[motor][param], label=param)
    #         ax.set_title(f"Motor: {motor}")
    #         ax.legend()
    #         ax.grid(True)

    # plt.pause(0.001)
    
    if ENABLE_TORQUE_READOUT:
        motor = motor_names[-1]  # For example, plot only the first motor
        for param in param_names:
            motor_data[motor][param].append(observation[f"{motor}.{param}"])
        plot_img = draw_plot(motor_data[motor])
        cv2.imshow(f"motor {motor} param {param}", plot_img)

    # Always show FPS in a small plot:
    fps_img = draw_plot({"FPS": list(fps_data)})  # Wrap fps_data as dict
    cv2.imshow("FPS Plot", fps_img)

    # --- Send action to robot ---
    robot.send_action(action)

    # --- Break on 'q' ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# === Cleanup ===
robot.disconnect()
teleop.disconnect()
cv2.destroyAllWindows()
plt.ioff()
plt.show()


