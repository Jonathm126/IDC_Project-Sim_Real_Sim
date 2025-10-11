from lerobot.processor import (
    ProcessorStepRegistry,
    ObservationProcessorStep
)

from lerobot.configs.types import PipelineFeatureType, PolicyFeature

import json
from PIL import Image
from dataclasses import dataclass, field

# gemini
from gemini.gemini_utils import call_gemini_robotics_er, compute_center_angle, GEMINI_ID
from google.genai import Client

@dataclass
@ProcessorStepRegistry.register("gemini_annotation_processor")
class GeminiAnnotateProcessorStep(ObservationProcessorStep):
    """Get an annotated observation using Gemini for the first frame of the environment observation."""
    
    client: Client
    prompt: str
    current_annotation: tuple[float, float, float] = field(init=False, default=None)

    def __post_init__(self):
        pass        # placeholder

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
                x_px, y_px, rotation_deg, annotated = compute_center_angle(res, 'front_point', 'rear_point', img)
                self.current_annotation = (float(x_px), float(y_px), float(rotation_deg))
            except Exception as e:
                raise RuntimeError(f"Failed to convert Gemini reply to geometry: {e}")

        # append the annotation to the env features
        x_px, y_px, rotation_deg = self.current_annotation
        return {**observation, 'x_px': float(x_px), 'y_px': float(y_px),'rotation_deg': float(rotation_deg)}

    def reset(self):
        # resets the annotation between episodes
        self.current_annotation = None

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
