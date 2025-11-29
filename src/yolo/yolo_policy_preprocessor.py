from lerobot.processor.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
    EnvTransition,
    PipelineFeatureType,
    PolicyFeature
)
from dataclasses import dataclass
from typing import List

@dataclass
@ProcessorStepRegistry.register(name="filter_env_processor")
class FilterEnvObsProcessorStep(ProcessorStep):
    """ Removes selected column indices from a batched feature tensor (BxD)
    feature_name (e.g. "observation.environment_state")
    columns_to_remove (list of indices)"""

    feature_name: str = None
    remove_indices: list[int] = None 
    
    def __post_init__(self):
        if self.remove_indices is None:
            self.remove_indices = []

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        tensor = transition[TransitionKey.OBSERVATION][self.feature_name]
        D = tensor.shape[1]
        keep_mask = [i for i in range(D) if i not in self.remove_indices]
        transition[TransitionKey.OBSERVATION][self.feature_name] = tensor[:, keep_mask]
        return transition

    def transform_features(self, features):
        return features

@dataclass
@ProcessorStepRegistry.register(name="remove_feature_processor")
class RemoveFeatureProcessorStep(ProcessorStep):
    """
    Removes a feature entirely from a transition.
    feature_name: str  
    """
    remove_feature_names: List[str] | None = None

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.remove_feature_names:
            return transition

        obs_key = TransitionKey.OBSERVATION.value
        obs = transition.get(obs_key) or {}

        return {
            **transition,
            obs_key: {k: v for k, v in obs.items() if k not in self.remove_feature_names},
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # nothing to remove
        if not self.remove_feature_names:
            return features

        obs = features.get(PipelineFeatureType.OBSERVATION, {})

        return {
                **features,
                PipelineFeatureType.OBSERVATION: {
                    k: v for k, v in obs.items() if k not in self.remove_feature_names
                },
        }