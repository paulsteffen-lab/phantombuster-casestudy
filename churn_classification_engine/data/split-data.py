from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split

from churn_classification_engine.config import settings

DATA_PATH = settings.data_dir / settings.data_filename
TRAIN_PATH = settings.data_dir / "train.csv"
TEST_PATH = settings.data_dir / "test.csv"
TARGET = "CHURN"
RANDOM_STATE = 42

if __name__ == "__main__":
    # Download the CSV file
    logger.info(f"Read CSV from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # First, split into Train (80%) and Temp (20%) [Validation + Test]
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=RANDOM_STATE
    )

    # Save the CSV file
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    logger.success(f"DataFrames saved at {settings.data_dir}")
