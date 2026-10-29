"""Configuration and constants for the Fake News Detection project."""

from pathlib import Path
import os
import random
import numpy as np

# Global Random Seed
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """Set global seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_CSV_PATH = RESULTS_DIR / "metrics.csv"

# Label definitions (As specified in workflow: Real = 1, Fake = 0)
LABEL_REAL = 1
LABEL_FAKE = 0
LABEL_MAP = {LABEL_REAL: "Real", LABEL_FAKE: "Fake"}

