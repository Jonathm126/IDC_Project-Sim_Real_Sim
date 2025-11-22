from lerobot.processor import (
    ProcessorStepRegistry,
    ObservationProcessorStep
)

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

import json
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple
from dataclasses import dataclass, field

# gemini
from gemini.gemini_utils import call_gemini_robotics_er, compute_center_angle, GEMINI_ID
from google.genai import Client

@dataclass
@ProcessorStepRegistry.register("gemini_annotation_processor")
class GeminiAnnotateProcessorStep(ObservationProcessorStep):
    """Get an annotated observation using Gemini for the first frame of the environment observation."""
    
    client: Optional[Client] = None
    prompt: Optional[str] = None
    manual_annotation: Optional[Tuple[float, float, float]] = None
    current_annotation: Optional[Tuple[float, float, float]] = field(init=False, default=None)
    debug: bool = False

    def __post_init__(self):
        if self.manual_annotation is not None:
            # Manual mode — skip requiring client and prompt
            self.current_annotation = self.manual_annotation
        else:
            # Automatic mode — require client and prompt
            if self.client is None:
                raise ValueError("`client` is required when `manual_annotation` is not provided.")
            if self.prompt is None:
                raise ValueError("`prompt` is required when `manual_annotation` is not provided.")

    def observation(self, observation):
        # call gemini if no previous data
        if self.current_annotation is None:
            # get top cam image
            try:
                img = observation['top_cam']
                img = Image.fromarray(img)
            except Exception as e:
                raise RuntimeError(f"Failed to process observation['top_cam']: {e}")

            # call gemini
            try:
                res = call_gemini_robotics_er(self.client, GEMINI_ID, img, self.prompt) #TODO do I need json parsing?
            except Exception as e:
                raise RuntimeError(f'Failed in Client call with {e}')

            # compute geometry
            try:
                res = json.loads(res)[0]
                x_norm, y_norm, theta_norm, annotated = compute_center_angle(res, 'front_point', 'rear_point', img)
                self.current_annotation = (float(x_norm), float(y_norm), float(theta_norm))
            except Exception as e:
                raise RuntimeError(f"Failed to convert Gemini reply {res} to geometry: {e}")

            # optionally display image
            if self.debug and annotated is not None:
                if isinstance(annotated, Image.Image):
                    annotated_np = np.array(annotated)
                    annotated_bgr = cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Gemini Annotation Debug View", annotated_bgr)
                    cv2.waitKey(1)
                else:
                    print("[DEBUG] Warning: Annotated image is not a PIL.Image")

        # append the annotation to the env features
        x_norm, y_norm, theta_norm = self.current_annotation
        return {**observation, 'x_px': float(x_norm), 'y_px': float(y_norm),'rotation_deg': float(theta_norm)}

    def reset(self):
        # resets the annotation between episodes
        self.current_annotation = self.manual_annotation

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
        # TODO if this is ok..
        return features
