from loguru import logger
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    fbeta_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from jinja2 import Template

from churn_classification_engine.config import settings

DATA_PATH = settings.data_dir / "test.csv"
PRED_PATH = settings.data_dir / "pred.csv"
REPORT_TEMPLATE_PATH = settings.reports_dir / "evaluation_template.html"
REPORT_PATH = settings.reports_dir / "evaluation_report.html"
POS_LABEL_THRESHOLD = 0.3


# Plot to base64 conversion utility
def plot_to_base64():
    """
    Converts a matplotlib plot to a base64-encoded PNG image.

    This function saves the current matplotlib plot to an in-memory buffer
    in PNG format, encodes the image data in base64, and returns the
    encoded string. The plot is saved with tight bounding box to minimize
    whitespace.
    """
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


def format_evaluation_data(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the true and predicted churn labels and format the data for evaluation.

    This function merges the true and predicted churn labels from the test set
    and formats the data for evaluation. It calculates the binarized predicted
    churn label using a threshold of 0.3, and assigns a risk level based on
    the predicted churn probability.

    Parameters:
        y_true (pd.DataFrame): The true churn labels.
        y_pred (pd.DataFrame): The predicted churn probabilities.

    Returns:
        pd.DataFrame: The formatted evaluation data.
    """
    # Merge the true and predicted churn labels
    df = y_true.merge(y_pred, left_index=True, right_index=True).sort_values(
        "CHURN_PROBA", ascending=False
    )

    # Calculate the binarized predicted churn label
    df["BINARIZED_Y_PRED"] = (df["CHURN_PROBA"] >= POS_LABEL_THRESHOLD).astype(int)

    # Assign a risk level based on the predicted churn probability
    df["RISK_LEVEL"] = pd.cut(
        df["CHURN_PROBA"],
        bins=[0, 0.1, 0.3, 0.5, 0.75, 1],
        labels=["no-risk", "low-risk", "moderate-risk", "risky", "high-risk"],
    )

    return df


def compute_group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute group metrics based on the risk level.

    This function computes the group size and accuracy based on the risk level
    for the predicted churn labels.

    Parameters:
        df (pd.DataFrame): The formatted evaluation data.

    Returns:
        pd.DataFrame: The group metrics.
    """
    grouped_df = df.groupby("RISK_LEVEL", observed=False)
    group_size_df = grouped_df.size().rename("Size").to_frame()
    group_accuracy_df = (
        grouped_df.apply(
            lambda x: (x["CHURN"] == x["BINARIZED_Y_PRED"]).mean(), include_groups=False
        )
        .rename("Accuracy")
        .to_frame()
    )
    groups_df = group_size_df.merge(
        group_accuracy_df, left_index=True, right_index=True
    )

    return groups_df


def get_clf_report(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    """
    Generate a classification report for the model predictions.

    This function generates a classification report for the model predictions
    using the true and predicted churn labels.

    Parameters:
        y_true (pd.Series): The true churn labels.
        y_pred (pd.Series): The predicted churn labels.

    Returns:
        pd.DataFrame: The classification report.
    """
    clf_report = (
        pd.DataFrame(
            classification_report(
                y_true, y_pred, target_names=["no-churn", "churn"], output_dict=True
            )
        )
        .drop(columns="accuracy")
        .T
    )

    return clf_report


if __name__ == "__main__":
    # Load the CSV files
    logger.info(f"Read CSV files from {settings.data_dir}")
    y_true = pd.read_csv(
        DATA_PATH, index_col="CUSTOMER_ID", usecols=["CHURN", "CUSTOMER_ID"]
    )
    y_pred = pd.read_csv(
        PRED_PATH, index_col="CUSTOMER_ID", usecols=["CHURN_PROBA", "CUSTOMER_ID"]
    )

    # Merge y_true and y_pred, and format the data for evaluation
    df = format_evaluation_data(y_true, y_pred)

    # Evaluate the model predictions
    f2_score = fbeta_score(df["CHURN"], df["BINARIZED_Y_PRED"], beta=2, pos_label=1)
    clf_report = get_clf_report(df["CHURN"], df["BINARIZED_Y_PRED"])
    groups_df = compute_group_metrics(df)

    # Generate ROC Curve
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        df["CHURN"], df["CHURN_PROBA"], plot_chance_level=True, despine=True
    )
    roc_img = plot_to_base64()
    plt.close()

    # Generate Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(
        df["CHURN"], df["CHURN_PROBA"], plot_chance_level=True, despine=True
    )
    pr_img = plot_to_base64()
    plt.close()

    # Render HTML template
    with open(REPORT_TEMPLATE_PATH) as f:
        template = Template(f.read())

    html_output = template.render(
        threshold=POS_LABEL_THRESHOLD,
        churners_distribution=groups_df.to_html(),
        f2_score=f2_score,
        classification_report=clf_report.to_html(),
        roc_curve=roc_img,
        pr_curve=pr_img,
    )

    # Saving report
    with open(REPORT_PATH, "w") as f:
        f.write(html_output)

    logger.success(f"Evaluation saved at {REPORT_PATH}")
