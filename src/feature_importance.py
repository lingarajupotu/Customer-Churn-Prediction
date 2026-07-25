import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(model, X_train):
    """
    Plot feature importance for tree-based models.
    """

    os.makedirs("results/plots", exist_ok=True)

    # Get feature names after preprocessing
    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    # Get trained model
    estimator = model.named_steps["model"]

    importances = estimator.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Save CSV
    importance_df.to_csv(
        "results/feature_importance.csv",
        index=False
    )

    # Plot Top 15
    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"][:15][::-1],
        importance_df["Importance"][:15][::-1]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importances")

    plt.tight_layout()

    plt.savefig(
        "results/plots/feature_importance.png"
    )

    plt.close()

    print("\nFeature importance saved successfully!")