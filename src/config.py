# ==========================================
# Project Configuration
# ==========================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5

TARGET_COLUMN = "Exited"

DROP_COLUMNS = [
    "RowNumber",
    "CustomerId",
    "Surname"
]

MODEL_PATH = "models/best_model.pkl"
