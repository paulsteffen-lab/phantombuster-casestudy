from loguru import logger
import pandas as pd
import joblib

from churn_classification_engine.model.utils import get_X_y
from churn_classification_engine.config import settings

INPUT_PATH = settings.data_dir / "test.csv"
OUTPUT_PATH = settings.data_dir / "pred.csv"
MODEL_PATH = settings.models_dir / "calibrated_hist_gradient_boosting.pkl"
TARGET = "CHURN"
RANDOM_STATE = 42

if __name__ == "__main__":
    # Load the CSV file
    logger.info(f"Read CSV from {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, index_col="CUSTOMER_ID")

    # Split the data into X and y
    X, _ = get_X_y(df)

    # Load the model
    model = joblib.load(MODEL_PATH)

    # Predict churn proba
    logger.info("Predicting churn proba in progress...")
    y_pred = model.predict_proba(X)[:, 1]

    # Save the prediction
    pd.Series(y_pred, index=X.index, name="CHURN_PROBA").to_csv(
        OUTPUT_PATH, header=True
    )
    logger.success(f"Predictions saved at {OUTPUT_PATH}")
