import numpy as np
import cv2
import json
import torch
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from ultralytics.utils.plotting import Annotator
from ultralytics.utils import ops

# define const
YOLO_SOURCE_NAMES = ['car', 'source']
YOLO_TARGET_NAMES = ['target']
YOLO_ANN_COLORS = {
    "target": (255, 140, 0),     # dark orange
    "source": (46, 139, 87),     # sea green
}

def yolo_preprocess(x):
    """Accepts torch.Tensor or np.ndarray (CHW or HWC).
    Returns HWC BGR uint8 numpy."""
    # ---- torch -> numpy ----
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 3 and x.shape[0] in (1,3):  # CHW → HWC
            x = x.permute(1, 2, 0)
        x = x.numpy()

    # ---- numpy ----
    if x.ndim == 3 and x.shape[0] in (1,3):      # CHW numpy → HWC
        x = np.transpose(x, (1,2,0))

    # ensure uint8
    if x.dtype != np.uint8:
        x = (np.clip(x, 0, 1) * 255).astype(np.uint8)

    # HWC RGB → BGR
    return x[..., ::-1]

# def yolo_preprocess_tensor(t): 
#     ''' pre-process to BGR PIL from Tensor ''' 
#     pil = to_pil_image(t.cpu()) 
#     rgb = np.array(pil) 
#     return rgb[..., ::-1]


def yolo_video_from_dataset(model, dataset, episode, fps, out_path):
    ''' used for pure YOLO inference on a Lerobot dataset '''
    # episode bounds
    ep = dataset.meta.episodes[episode]
    start = ep["dataset_from_index"]
    end   = ep["dataset_to_index"]

    # read first frame for size
    frame = dataset[start]['observation.images.top_cam']
    rgb = yolo_preprocess(frame)
    h, w = rgb.shape[:2]

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # iterate episode frames only
    for idx in tqdm(range(start, end + 1)):
        frame = dataset[idx]['observation.images.top_cam']

        rgb = yolo_preprocess(frame)

        # YOLO inference (RGB in)
        results = model.predict(rgb, verbose=False)

        # annotated output (BGR)
        annotated_bgr = results[0].plot()
        writer.write(annotated_bgr)

    writer.release()
    return out_path

def yolo_draw_single_obb(img, box, label, conf = None):
    """
    Draw a single OBB box on a PIL image using a color based on label.

    Args:
        img: PIL.Image
        box: tensor/list of shape (5,) in xywhr pixel-space [cx,cy,w,h,theta]
        label: "target" or "source"

    Returns:
        Annotated PIL image
    """
    assert label in YOLO_ANN_COLORS, f"Invalid label: {label}"
    assert box.shape == (5,), "box must be single xywhr (cx,cy,w,h,r)"
    assert (box[:4] > 1).all(), "xywh must be in pixel units, not normalized"

    # convert xywhr → 4 corners
    xyxyxyxy = ops.xywhr2xyxyxyxy(box)  # shape (4,)

    # prepare text
    cx, cy, w, h, theta = box.tolist()
    text = f"{label} x:{cx:.2f} y:{cy:.2f} r:{theta:.2f}"
    if conf is not None:
        text += f" conf:{conf:.2f}"

    ann = Annotator(img.copy(), pil=True)
    ann.box_label(xyxyxyxy, text, YOLO_ANN_COLORS[label])

    # cast to PIL
    res_img = ann.result()
    if isinstance(res_img, np.ndarray):
        res_img = Image.fromarray(res_img)
    return res_img

def yolo_draw_center_orientation(img, x, y, r, color=(1, 0, 0), scale=40):
    """
    Draws center + orientation arrow using Ultralytics Annotator.
    Args:
        img: PIL.Image
        x, y: pixel coords
        r: angle in radians
        color: RGB tuple
        scale: arrow length
    """
    # ignore missing detections
    if x < 0 or y < 0:
        return img
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # draw circle
    cx, cy = int(x), int(y)
    cv2.circle(frame, (cx, cy), 8, color, thickness=-1, lineType=cv2.LINE_AA)
    # only draw orientation if r is provided (not None)
    if r is not None:
        ex = int(cx + scale * np.cos(r))
        ey = int(cy + scale * np.sin(r))

        cv2.arrowedLine(
            frame,
            (cx, cy),
            (ex, ey),
            color,
            thickness=8,
            tipLength=0.25,
            line_type=cv2.LINE_AA
        )
        # --- Add angle text ---
        text = f"{r:.2f} rad"
        cv2.putText(
            frame,
            text,
            (cx + 10, cy - 10),          # offset from center
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )
    # to PIL
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

def yolo_map_class_name(name):
    '''
    Map raw YOLO class names into canonical labels.

    Args:
        name (str): Original YOLO class name.
        target_names (set[str]): Names considered as "target".
        source_names (set[str]): Names considered as "source".

    Returns:
        str: "target" or "source".
    '''
    if name in YOLO_TARGET_NAMES:
        return "target"
    if name in YOLO_SOURCE_NAMES:
        return "source"
    raise ValueError(f"Class name '{name}' not in target or source lists.")

def yolo_postprocess_res(res):
    """
    Selects the best OBB per class, returns normalized (x,y,r) values,
    and draws each chosen box on the original image.

    Args:
        res: YOLO Results object with .obb, .names, .orig_img.

    Returns:
        annotation vector: [source_x, source_y, source_r, target_x, target_y, target_r]
        PIL image with drawn OBBs.
    """
    obb = res.obb
    names = res.names
    ann = Image.fromarray(res.orig_img[..., ::-1].copy())

    # init defaults
    ordered_classes = ["source", "target"]
    out = {k: {"x": -1.0, "y": -1.0, "r": -1.0} for k in ordered_classes}

    # if there are no detections at all return the default
    if obb is None or obb.data.numel() == 0:
        vec = [out[k][c] for k in ordered_classes for c in ("x","y","r")]
        return vec, ann

    # normalize the coordiantes to 0..1
    cls, conf = obb.cls.long(), obb.conf
    x, y, _, _, theta = obb.xywhr.unbind(1)
    H, W = obb.orig_shape
    x_n, y_n = x / W, y / H

    # scan detections
    for cid, cname in names.items():
        key = yolo_map_class_name(cname)
        mask = (cls == cid)

        if mask.any():
            idx = torch.argmax(conf[mask])
            ridx = torch.nonzero(mask, as_tuple=True)[0][idx]
            out[key]["x"] = x_n[ridx].item()
            out[key]["y"] = y_n[ridx].item()
            out[key]["r"] = theta[ridx].item()

            # annotate image
            ann = yolo_draw_single_obb(ann, obb.xywhr[ridx], key, obb.conf[ridx])

    # flatten
    vec = [out[k][c] for k in ordered_classes for c in ("x","y","r")]
    return vec, ann

def yolo_annotation_from_json(json_path, frame_index):
    """
    Load YOLO annotation vec for a single frame from an episode JSON file.

    Args:
        json_path: Path to ep_XXX.json
        frame_index: int dataset index

    Returns:
        vec list: [sx, sy, sr, tx, ty, tr]  or None if not found
    """
    with open(json_path, "r") as f:
        ann = json.load(f)

    for rec in ann:
        if rec["frame_index"] == frame_index:
            return rec["vec"]

    return None
