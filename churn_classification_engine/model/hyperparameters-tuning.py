from loguru import logger
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from skrub import ToCategorical, MinHashEncoder, TableVectorizer
from sklearn.metrics import make_scorer, fbeta_score

from churn_classification_engine.config import settings, hyperparameters_search_area

DATA_PATH = settings.data_dir / "train.csv"
HYPERPARAMETERS_DB_PATH = f"sqlite:///{settings.data_dir}/optuna_study.db"
TARGET = "CHURN"
RANDOM_STATE = 42

def objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """
    Optimize hyperparameters for a machine learning pipeline using Optuna.

    This function defines an Optuna objective for hyperparameter optimization
    of a machine learning pipeline consisting of a TableVectorizer and a
    HistGradientBoostingClassifier. It evaluates the model using cross-validation
    with the F2 score as the metric.

    Parameters:
        trial (optuna.trial.Trial): The trial object for suggesting hyperparameters.
        X (pd.DataFrame): The input features for the model.
        y (pd.Series): The target variable.

    Returns:
        float: The mean F2 score from cross-validation, negated for minimization.
    """
    # Define the hyperparameters to optimize
    n_components = trial.suggest_int(**hyperparameters_search_area.n_components)
    
    learning_rate = trial.suggest_float(**hyperparameters_search_area.learning_rate)
    max_iter = trial.suggest_int(**hyperparameters_search_area.max_iter)
    max_depth = trial.suggest_int(**hyperparameters_search_area.max_depth)
    min_samples_leaf = trial.suggest_int(**hyperparameters_search_area.min_samples_leaf)
    max_bins = trial.suggest_int(**hyperparameters_search_area.max_bins)

    # Set up the pipeline
    pipeline = Pipeline(steps=[
        ('tablevectorizer',
         TableVectorizer(high_cardinality=MinHashEncoder(n_components=n_components),
                         low_cardinality=ToCategorical())),
        ('histgradientboostingclassifier',
         HistGradientBoostingClassifier(
             class_weight='balanced',
             learning_rate=learning_rate,
             max_iter=max_iter,
             max_depth=max_depth,
             min_samples_leaf=min_samples_leaf,
             max_bins=max_bins,
             random_state=42,
         ))
    ])

    # Use F2 score as the scoring metric
    f2_scorer = make_scorer(fbeta_score, beta=2)

    # Use cross-validation to evaluate the model
    cv_scores = cross_val_score(pipeline, 
                                X, 
                                y, 
                                cv=StratifiedKFold(n_splits=5), 
                                scoring=f2_scorer, 
                                n_jobs=5)
    
    # Return the mean f2 score
    return cv_scores.mean()

if __name__ == "__main__":
    # Load the CSV file
    logger.info(f"Read CSV from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, index_col="CUSTOMER_ID")
    
    # Split the data into X and y
    X = df.drop(columns="CHURN")
    y = df["CHURN"]
    
    # Create and optimize the study
    study = optuna.create_study(study_name="hyperparameters_tuning",
                                storage=HYPERPARAMETERS_DB_PATH, 
                                direction='maximize',
                                load_if_exists=True)
    study.optimize(lambda trial: objective(trial, X, y), n_trials=100, show_progress_bar=True)

    # Log the best hyperparameters and the best score found
    logger.success(f"Best hyperparameters found: {study.best_params}")
    logger.success(f"Best cross-validation F2 score: {study.best_value}")
