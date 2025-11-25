import json
import numpy as np
import cv2

from PIL import Image
from typing import Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

# lerobot
from lerobot.processor import (
    ProcessorStepRegistry,
    ObservationProcessorStep
)
from lerobot.configs.types import PipelineFeatureType, PolicyFeature

# yolo
from yolo_utils import yolo_preprocess_tensor, yolo_postprocess_res
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
            rgb = yolo_preprocess_tensor(img)
            res = self.model.predict(rgb, conf=0.5, verbose=False)[0] # predict
            vec, _ = yolo_postprocess_res(res)   # vec = [sx,sy,sr, tx,ty,tr]
        except Exception as e:
            raise RuntimeError(f"Failed to predict: {e}")

        # parse
        assert len(vec) == 6
        sx, sy, sr, tx, ty, tr  = map(float, vec)
        assert all((0 <= v <= 1.0 or v == -1.0) for v in (sx, sy, tx, ty))
        assert all((-np.pi/2 <= v <= np.pi/2) for v in (sr, tr))

        # return scalar fields
        return {
            **observation,
            "source_x": sx,
            "source_y": sy,
            "source_r": sr,
            "target_x": tx,
            "target_y": ty,
            "target_r": tr,
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
        features[PipelineFeatureType.OBSERVATION]["observation.environment_state"] = PolicyFeature(
            dtype="float32",
            shape=[6],
            names=[
                "source_x", "source_y", "source_r",
                "target_x", "target_y", "target_r",
            ],
        )
        return features
