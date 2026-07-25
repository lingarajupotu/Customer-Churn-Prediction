# ==========================================
# Customer Churn Prediction - Model Training
# ==========================================

# Standard Library
import warnings

warnings.filterwarnings("ignore")
import os
import joblib
from tuning import tune_random_forest
from visualize import create_visualizations
from feature_importance import plot_feature_importance
from trainer import train_models
from sklearn.model_selection import train_test_split
from cross_validation import run_cross_validation
from sklearn.pipeline import Pipeline

from config import *
from data_loader import load_data
from preprocessing import create_preprocessor
from models import get_models
from evaluate import evaluate_model
def main():
    # ==========================================
    # Load Dataset
    # ==========================================
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    df = load_data()

    print("Dataset Loaded Successfully")
    print(f"Dataset Shape: {df.shape}")

    # ==========================================
    # Drop Unnecessary Columns
    # ==========================================

    df.drop(columns=DROP_COLUMNS, inplace=True)

    print("Unnecessary columns removed.")

    # ==========================================
    # Separate Features and Target
    # ==========================================

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    print(f"Features Shape : {X.shape}")
    print(f"Target Shape   : {y.shape}")

    # ==========================================
    # Train-Test Split
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain-Test Split Completed")

    print(f"Training Samples : {X_train.shape[0]}")
    print(f"Testing Samples  : {X_test.shape[0]}")

    # ==========================================
    # Identify Numerical and Categorical Columns
    # ==========================================

    preprocessor = create_preprocessor(X_train)

    print("Preprocessor Created Successfully")

    # ==========================================
    # Define Machine Learning Models
    # ==========================================

    models = get_models()


    # ==========================================
    # Create ML Pipelines
    # ==========================================

    pipelines = {}


    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipelines[name] = pipeline


    print("All Pipelines Created Successfully")
    results_df, best_model, best_model_name, best_accuracy = train_models(
        pipelines,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n")
    print(results_df)

    print("\nBest Model :", best_model_name)
    print(f"Best Accuracy : {best_accuracy:.4f}")

    cv_df = run_cross_validation(
        pipelines,
        X,
        y,
        CV_FOLDS
    )

    print(cv_df)
    cv_df.to_csv(
        "results/cross_validation.csv",
        index=False
    )
    # Save results
    results_df.to_csv(
        "results/model_comparison.csv",
        index=False
    )

    cv_df.to_csv(
        "results/cross_validation.csv",
        index=False
    )

    print("\nResults saved successfully.")

    # ==========================================
    # Save Best Model
    # ==========================================

    joblib.dump(best_model, MODEL_PATH)

    print(f"\nBest model saved at: {MODEL_PATH}")
    # ==========================================
    # Hyperparameter Tuning
    # ==========================================

    print("\n")
    print("=" * 60)
    print("Hyperparameter Tuning - Random Forest")
    print("=" * 60)

    grid_search = tune_random_forest(
        preprocessor,
        X_train,
        y_train
    )

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(f"\nBest Cross Validation Accuracy: {grid_search.best_score_:.4f}")

    best_rf = grid_search.best_estimator_

    metrics = evaluate_model(
        best_rf,
        X_test,
        y_test
    )

    print("\nPerformance of Tuned Random Forest")
    print("-" * 40)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    if metrics["Accuracy"] > best_accuracy:

        best_model = best_rf
        best_accuracy = metrics["Accuracy"]
        best_model_name = "Random Forest (Tuned)"

        joblib.dump(best_model, MODEL_PATH)

    print("Creating Visualizations...")

    create_visualizations(
        best_model,
        X_test,
        y_test
    )
    # Plot feature importance only for tree-based models
    estimator = best_model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        plot_feature_importance(best_model, X_train)

        
if __name__ == "__main__":
    main()
