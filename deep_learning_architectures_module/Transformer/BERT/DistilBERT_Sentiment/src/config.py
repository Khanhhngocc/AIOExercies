from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_NAME = "cornell-movie-review-data/rotten_tomatoes"
MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 128
SEED = 59

CHECKPOINT_DIR = PROJECT_ROOT / "distilbert_sentiment_model"
MODEL_DIR = PROJECT_ROOT / "distilbert_sentiment_best"
RESULTS_DIR = PROJECT_ROOT / "results"