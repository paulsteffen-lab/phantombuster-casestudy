import pandas as pd
from typing import Tuple

def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the data into X and y.

    This function splits the input DataFrame into the feature matrix X
    and the target vector y.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: The feature matrix X and the target vector y.
    """
    X = df.drop(columns=["CHURN", "COUNTRY_CODE"])
    y = df["CHURN"]

    return X, y