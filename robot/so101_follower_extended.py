import time
import logging
from typing import Any
from functools import cached_property
from dataclasses import dataclass, field
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.robots.config import RobotConfig

logger = logging.getLogger(__name__)

## this subclass extends the SO101 follower to return additional parameters to it's obervation dict

@RobotConfig.register_subclass("so101_follower_ext")
@dataclass
class SO101FollowerExtConfig(SO101FollowerConfig):
    # Map register -> feature suffix, e.g. {"Present_Current":"current"}
    extra_motor_regs: dict[str, str] = field(default_factory=dict)

class SO101FollowerExt(SO101Follower):
    """
    Extends SO101Follower to include extra per-motor registers in the observation.
    Configure `extra_motor_regs` as { "RegisterName": "feature_suffix" }.
    Example: { "Present_Current": "current", "Present_Velocity": "vel" }
    """
    def __init__(self, config: SO101FollowerExtConfig):
        super().__init__(config)
        self.extra_motor_regs = dict(config.extra_motor_regs)

    # --- override properties ---
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        # start with parent definition
        features = dict(super().observation_features)
        # extra motor features
        for _, suffix in self.extra_motor_regs.items():
            features.update({f"{m}.{suffix}": float for m in self.bus.motors})
        return features

    
    # --- override methods ---
    def get_observation(self) -> dict[str, Any]:
        obs_dict = super().get_observation()
        
        # add extra obs
        for reg, suffix in self.extra_motor_regs.items():
            t0 = time.perf_counter()
            vals = self.bus.sync_read(reg)
            obs_dict.update({f"{m}.{suffix}": v for m, v in vals.items()})
            dt_ms = (time.perf_counter() - t0) * 1e3
            logger.debug(f"{self} read {reg}: {dt_ms:.1f}ms")
        return obs_dict