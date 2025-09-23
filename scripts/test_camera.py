from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation
import cv2
import os
import time
import numpy as np
from collections import deque

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_DECODING"] = "0"
os.environ["OPENCV_VIDEOIO_MSMF_FORCE_BGR"] = "1"

print("started")

# Camera configs
config0 = OpenCVCameraConfig(
    index_or_path=0,
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION,
)
config1 = OpenCVCameraConfig(
    index_or_path=1,
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION,
)

# Instantiate cameras
camera0 = OpenCVCamera(config0)
camera1 = OpenCVCamera(config1)
camera0.connect()
camera1.connect()

print("connected")

# FPS measurement
frame_count = 0
start_time = time.time()
fps = 0.0

# Graph buffers
fps_history = deque(maxlen=100)  # last 100 points

def draw_fps_graph(history, width=400, height=200, max_fps=60):
    graph = np.zeros((height, width, 3), dtype=np.uint8)

    if len(history) > 1:
        # Scale data
        pts = np.array([
            (int(x * width / len(history)),
             height - int(y * height / max_fps))
            for x, y in enumerate(history)
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(graph, [pts], isClosed=False, color=(0, 255, 0), thickness=2)

    # Add labels
    cv2.putText(graph, f"FPS Graph", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(graph, f"{history[-1]:.2f} FPS" if history else "",
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 1)

    return graph

try:
    while True:
        frame0 = camera0.async_read(timeout_ms=200)
        frame1 = camera1.async_read(timeout_ms=200)

        frame0_bgr = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
        frame1_bgr = cv2.cvtColor(frame1, cv2.COLOR_RGB2BGR)

        # Ensure same height for side-by-side concat
        if frame0_bgr.shape[0] != frame1_bgr.shape[0]:
            h = min(frame0_bgr.shape[0], frame1_bgr.shape[0])
            frame0_bgr = cv2.resize(frame0_bgr, (frame0_bgr.shape[1], h))
            frame1_bgr = cv2.resize(frame1_bgr, (frame1_bgr.shape[1], h))

        combined = np.hstack((frame0_bgr, frame1_bgr))

        # Update FPS
        frame_count += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            fps_history.append(fps)
            frame_count = 0
            start_time = time.time()

        # Overlay FPS on video
        cv2.putText(
            combined,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Draw FPS graph
        graph = draw_fps_graph(fps_history)

        # Show windows
        cv2.imshow("Live Camera Feed (0 & 1)", combined)
        cv2.imshow("FPS Graph", graph)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera0.disconnect()
    camera1.disconnect()
    cv2.destroyAllWindows()
