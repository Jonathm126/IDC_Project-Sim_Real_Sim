from pathlib import Path

# Assumes this file is somewhere inside the repo
REPO_ROOT = Path(__file__).resolve()
while not (REPO_ROOT / ".git").exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent

CALIBS_DIR = REPO_ROOT / "robot" / "calibrations"
MODELS_DIR = REPO_ROOT / "models"
DATASETS_DIR = REPO_ROOT / "datasets"
POLICIES_DIR = REPO_ROOT / "policies"
OUTPUTS_DIR = REPO_ROOT / "outputs"
EVAL_DIR = REPO_ROOT / "eval"

HF_NAME = 'jonathm126'