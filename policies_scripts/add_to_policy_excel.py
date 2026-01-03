import json
import sys
from pathlib import Path
import csv

if len(sys.argv) != 2:
    print("Usage: python add_to_policy_excel.py <policy_folder>")
    sys.exit(1)

policy_root = Path(sys.argv[1]).resolve()

# Find train configs under last/**/
configs = sorted(policy_root.rglob("last/**/train_config*.json"))

if not configs:
    print("No train_config*.json found under last/**/")
    sys.exit(1)

writer = csv.writer(sys.stdout)

# Excel header
writer.writerow([
    "Run",
    "Num Input Features",
    "Input Features (name:shape)",
    "Action Shape",
    "Model Dim",
    "Encoder Layers",
    "Decoder Layers",
    "Use VAE",
    "Latent Dim",
    "KL Weight",
    "Chunk Size",
    "N Action Steps",
    "Training Steps",
    "Temporal Ensemble",
])

for cfg_path in configs:
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    policy = cfg["policy"]
    inputs = policy["input_features"]

    run_name = policy.get("repo_id") or cfg_path.parents[2].name

    # name:[shape]
    feature_desc = [
        f"{name}:{spec.get('shape')}"
        for name, spec in inputs.items()
    ]

    writer.writerow([
        run_name,
        len(inputs),
        "; ".join(feature_desc),
        policy["output_features"]["action"]["shape"],
        policy["dim_model"],
        policy["n_encoder_layers"],
        policy["n_decoder_layers"],
        policy["use_vae"],
        policy["latent_dim"],
        policy["kl_weight"],
        policy["chunk_size"],
        policy["n_action_steps"],
        cfg["steps"],
        policy.get("temporal_ensemble_coeff") is not None,
    ])
