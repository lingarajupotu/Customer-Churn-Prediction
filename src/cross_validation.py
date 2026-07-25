import pandas as pd
from sklearn.model_selection import cross_val_score


def run_cross_validation(
    pipelines,
    X,
    y,
    cv_folds
):
    """
    Run cross validation for all models.
    """

    print("\n")
    print("=" * 60)
    print("Cross Validation Results")
    print("=" * 60)

    cv_results = []

    for name, pipeline in pipelines.items():

        scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=cv_folds,
            scoring="accuracy",
            n_jobs=-1
        )

        cv_results.append({
            "Model": name,
            "Mean Accuracy": scores.mean(),
            "Standard Deviation": scores.std()
        })

    cv_df = pd.DataFrame(cv_results)

    cv_df = cv_df.sort_values(
        by="Mean Accuracy",
        ascending=False
    ).reset_index(drop=True)

    return cv_df