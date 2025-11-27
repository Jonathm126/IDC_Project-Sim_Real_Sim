from lerobot.processor.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
    EnvTransition
)
from dataclasses import dataclass

@dataclass
@ProcessorStepRegistry.register(name="filter_env_processor")
class FilterEnvObsProcessorStep(ProcessorStep):
    """ Removes selected column indices from a batched feature tensor (B×D)
    feature_name (e.g. "observation.environment_state")
    columns_to_remove (list of indices)"""

    feature_name: str = "observation.environment_state"
    remove_indices: list[int] = None   # e.g. [2, 5]
    
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