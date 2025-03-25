from loguru import logger
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from skrub import ToCategorical, MinHashEncoder, TableVectorizer

from churn_classification_engine.config import settings

DATA_PATH = settings.data_dir / "train.csv"
MODEL_PATH = settings.models_dir / "calibrated_hist_gradient_boosting.pkl"
TARGET = "CHURN"
RANDOM_STATE = 42

if __name__ == "__main__":
    # Load the CSV file
    logger.info(f"Read CSV from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, index_col="CUSTOMER_ID")

    # Split the data into X and y
    X = df.drop(columns=["CHURN", "COUNTRY_CODE"])
    y = df["CHURN"]

    # Instantiate the model
    pipeline = Pipeline(
        steps=[
            (
                "tablevectorizer",
                TableVectorizer(
                    #high_cardinality=MinHashEncoder(n_components=35),
                    low_cardinality=ToCategorical(),
                ),
            ),
            (
                "histgradientboostingclassifier",
                HistGradientBoostingClassifier(
                    class_weight="balanced",
                    learning_rate=0.01703384477761088,
                    max_iter=195,
                    max_depth=11,
                    min_samples_leaf=50,
                    max_bins=255,
                    random_state=42,
                ),
            ),
        ]
    )

    # Train the model
    logger.info("Training and calibration in progress...")
    pipeline.fit(X, y)

    # Calibrate the model
    calibrated_pipeline = CalibratedClassifierCV(
        FrozenEstimator(pipeline), method="sigmoid"
    )
    calibrated_pipeline.fit(X, y)

    # Save the model
    joblib.dump(calibrated_pipeline, MODEL_PATH)
    logger.success(f"Model saved at {MODEL_PATH}")
