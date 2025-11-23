from PIL import Image
from google.genai import types, Client
import math
from PIL import ImageDraw, Image

GEMINI_ID = "gemini-robotics-er-1.5-preview"

# Resizing to speed-up rendering
def get_image_resized(img_path):
    img = Image.open(img_path)
    img = img.resize(
        (800, int(800 * img.size[1] / img.size[0])), Image.Resampling.LANCZOS
    )
    return img


def call_gemini_robotics_er(client: Client, model, img, prompt, config=None, print_response = False):
    if config is None:
        config = types.GenerateContentConfig(
            temperature=0.5, thinking_config=types.ThinkingConfig(thinking_budget=0)
        )

    image_response = client.models.generate_content(
            model=model,
            contents=[img, prompt],
            config=config,
    )

    if print_response:
        print(image_response.text)

    return parse_json(image_response.text)

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            # Remove everything before "```json"
            json_output = "\n".join(lines[i + 1 :])
            # Remove everything after the closing "```"
            json_output = json_output.split("```")[0]
            break  # Exit the loop once "```json" is found
    return json_output

def compute_center_angle(data: dict, prompt1: str, prompt2: str, img: Image):
    """
    Compute center (x_px, y_px) and rotation angle (deg)
    from Gemini output with normalized coords [0..1].
    """
    h, w = img.height, img.width

    #  Gemini gives coordinates in range [1, 1000]
    fy_n, fx_n = data[prompt1]
    ry_n, rx_n = data[prompt2]

    # center (normalized)
    cx_n, cy_n = (fx_n + rx_n) / 2000.0, (fy_n + ry_n) / 2000.0

    # rotation (deg, 0°=left, CW+)
    dx_n, dy_n = fx_n - rx_n, fy_n - ry_n
    theta_deg = (math.degrees(math.atan2(dy_n, -dx_n)) % 360)
    theta_norm = theta_deg / 360.0

    # drawing
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    r = 6

    # Convert normalized coordinates to pixel values for drawing
    fx_px, fy_px = fx_n * w / 1000, fy_n * h / 1000
    rx_px, ry_px = rx_n * w / 1000, ry_n * h / 1000
    cx_px, cy_px = cx_n * w, cy_n * h

    draw.ellipse((fx_px - r, fy_px - r, fx_px + r, fy_px + r), fill="red", outline="black")   # front
    draw.ellipse((rx_px - r, ry_px - r, rx_px + r, ry_px + r), fill="blue", outline="black")  # rear
    draw.ellipse((cx_px - r, cy_px - r, cx_px + r, cy_px + r), fill="green", outline="black") # center
    draw.line((fx_px, fy_px, rx_px, ry_px), fill="blue", width=3)
    draw.text((cx_px + 8, cy_px - 12), f"θ={theta_deg:.1f}°", fill="green")

    return cx_n, cy_n, theta_norm, annotated