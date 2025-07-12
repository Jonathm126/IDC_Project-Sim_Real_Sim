from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation
import cv2

print("started")

# Construct an `OpenCVCameraConfig` with your desired FPS, resolution, color mode, and rotation.
config = OpenCVCameraConfig(
    index_or_path=2,
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

# Instantiate and connect an `OpenCVCamera`, performing a warm-up read (default).
camera = OpenCVCamera(config)
camera.connect()

print("connected")

# Read frames asynchronously in a loop via `async_read(timeout_ms)`
try:
    while True:
        frame = camera.async_read(timeout_ms=200)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Live Camera Feed", frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.disconnect()
    cv2.destroyAllWindows()