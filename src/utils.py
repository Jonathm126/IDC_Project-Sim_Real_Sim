from IPython.display import display, HTML
import pprint
import torch
import numpy as np

def scroll_print(data): 
    display(HTML(f"""<div style="max-height:300px; overflow:auto; border:1px solid #ccc; padding:10px;">
    <pre>{pprint.pformat(data, indent=2, width=80)}</pre>
    </div>"""))

def process_obs_to_np(obs):
    return {
    k: (
        (
            v.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
            if v.dtype == torch.uint8
            else (v.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        )
        if "images" in k
        else v.detach().cpu().squeeze(0).float().numpy()
    )
    for k, v in obs.items()
}