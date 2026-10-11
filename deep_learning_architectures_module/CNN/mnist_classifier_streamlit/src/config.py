from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "lenet_mnist.pt"

HISTORY_PATH = LOG_DIR / "history.csv"
CURVE_PATH = LOG_DIR / "learning_curves.png"


# ============================================================
# TRAINING CONFIG
# ============================================================

SEED = 42

TRAIN_RATIO = 0.9

BATCH_SIZE = 256

NUM_EPOCHS = 10

LEARNING_RATE = 1e-3

NUM_CLASSES = 10