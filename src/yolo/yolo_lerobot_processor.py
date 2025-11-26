import numpy as np

from pathlib import Path
from dataclasses import dataclass

# lerobot
from lerobot.processor import (
    ProcessorStepRegistry,
    ObservationProcessorStep
)
from lerobot.configs.types import PipelineFeatureType, PolicyFeature, FeatureType

# yolo
from src.yolo.yolo_utils import yolo_preprocess, yolo_postprocess_res, yolo_draw_center_orientation, YOLO_ANN_COLORS
from ultralytics import YOLO

@dataclass
@ProcessorStepRegistry.register("yolo_annotation_processor")
class YoloAnnotateProcessorStep(ObservationProcessorStep):
    """Get an the environment state vector which includes the position and orientation (x,y,r) of the source and target items.
    Uses a pre-trained YOLO sub-model for this annotation.
    Inputs: path to yolo model (.pt)"""
    model_path: Path
    cam_name: str

    def __post_init__(self):
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to build YOLO model: {e}")

    def observation(self, observation):
        try:
            img = observation[self.cam_name]
        except Exception as e:
            raise RuntimeError(f"Failed to process observation: {e}")

        # annotate
        try:
            bgr = yolo_preprocess(img)
            res = self.model.predict(bgr, conf=0.5, verbose=False)[0] # predict
            vec, _ = yolo_postprocess_res(res)   # vec = [sx,sy,sr, tx,ty,tr]
        except Exception as e:
            raise RuntimeError(f"Failed to predict: {e}")

        # parse
        assert len(vec) == 6
        sx, sy, sr, tx, ty, tr  = map(float, vec)
        assert all((0 <= v <= 1.0) or v == -1.0 for v in (sx, sy, tx, ty))
        assert all((-np.pi/2 <= v <= np.pi/2) for v in (sr, tr))
        
        # annotate
        H,W = img.shape[:2]
        ann = yolo_draw_center_orientation(img.copy(), sx*W, sy*H, sr, YOLO_ANN_COLORS["source"])
        ann = yolo_draw_center_orientation(ann, tx*W, ty*H, tr, YOLO_ANN_COLORS["target"])

        # return scalar fields
        return {
            **observation,
            "source_x": sx,
            "source_y": sy,
            "source_r": sr,
            "target_x": tx,
            "target_y": ty,
            "target_r": tr,
            f"{self.cam_name}_bbox": np.array(ann)
        }

    def reset(self):
        pass

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, any]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the feature dict to reflect the added observation.environment_state.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary.
        """
        # Add our new env-state feature
        for name in ["source_x", "source_y", "source_r",
                    "target_x", "target_y", "target_r"]:
            features[PipelineFeatureType.OBSERVATION][name] = PolicyFeature(
                type = FeatureType.ENV,
                shape = (1,)
            )
        return features
