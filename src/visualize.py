import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)


def create_visualizations(model, X_test, y_test):
    """
    Generate and save evaluation plots.
    """

    os.makedirs("results/plots", exist_ok=True)

    # ----------------------------
    # Confusion Matrix
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap="Blues",
        ax=ax
    )

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/plots/confusion_matrix.png")
    plt.close()

    # ----------------------------
    # ROC Curve
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax
    )

    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig("results/plots/roc_curve.png")
    plt.close()

    # ----------------------------
    # Precision-Recall Curve
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax
    )

    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig("results/plots/precision_recall_curve.png")
    plt.close()

    print("\nEvaluation plots saved successfully!")