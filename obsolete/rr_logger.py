import time
import os
from pathlib import Path
import numpy as np
import torch
import rerun as rr


class RRLogger:
    def __init__(self, 
                application_id: str,
                recording_id  : str | None = None,
                blueprint_path: Path | None = None):
        '''
        application_id (str): Identifier for the Rerun application session.
        recording_id (str | None, default=None): Unique recording identifier. If not provided,
            defaults to the current UNIX timestamp.
        blueprint_path (Path | None, default=None): Path to a saved Rerun blueprint JSON file.
            If supplied and exists, it is loaded via rr.blueprint.Blueprint and applied.
        '''
        self.start_time = time.time()
        self.frame_count = 0

        # Default recording ID to timestamp if not supplied
        if recording_id is None:
            recording_id = str(int(self.start_time))

        # Init rerun
        batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
        os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
        rr.init(application_id=application_id, recording_id=recording_id)
        memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
        rr.spawn(memory_limit=memory_limit)

        # Load blueprint if provided
        if blueprint_path:
            raise NotImplementedError
        
    def _to_numpy(self, val):
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu()
            if val.ndim == 4:  # assume (B, C, H, W)
                val = val.squeeze(0).permute(1, 2, 0)
            return val.numpy()
        return np.array(val)

    def log(self, observation: dict, action: dict, scalars: dict = None):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        # Scalars
        if scalars:
            for k, v in scalars.items():
                rr.log(f"scalars/{k}", rr.Scalar(float(v)))

        # Observations
        for obs, val in observation.items():
            arr = self._to_numpy(val)
            if arr.ndim in (2, 3):  # image
                rr.log(f"observation/{obs}", rr.Image(arr.astype(np.uint8)), static=True)
            elif arr.ndim == 1:
                for i, v in enumerate(arr):
                    rr.log(f"observation/{obs}/{i}", rr.Scalar(float(v)))
            else:
                rr.log(f"observation/{obs}", rr.Scalar(float(arr)))

        # Actions
        for act, val in action.items():
            arr = self._to_numpy(val)
            if arr.ndim == 1:
                for i, v in enumerate(arr):
                    rr.log(f"action/{act}/{i}", rr.Scalar(float(v)))
            else:
                rr.log(f"action/{act}", rr.Scalar(float(arr)))

        # FPS
        rr.log("system/fps", rr.Scalar(fps))
