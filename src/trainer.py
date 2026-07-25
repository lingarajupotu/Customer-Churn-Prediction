from evaluate import evaluate_model, results_dataframe


def train_models(
    pipelines,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train all models and evaluate them.
    """

    results = []

    best_model = None
    best_model_name = None
    best_accuracy = 0

    for name, pipeline in pipelines.items():

        print("=" * 60)
        print(f"Training {name}...")
        print("=" * 60)

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(
            pipeline,
            X_test,
            y_test
        )

        results.append({
            "Model": name,
            **metrics
        })

        print(f"Accuracy : {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall   : {metrics['Recall']:.4f}")
        print(f"F1 Score : {metrics['F1 Score']:.4f}")
        print(f"ROC AUC  : {metrics['ROC AUC']:.4f}")

        if metrics["Accuracy"] > best_accuracy:
            best_accuracy = metrics["Accuracy"]
            best_model = pipeline
            best_model_name = name

    print("\nTraining Completed Successfully!")

    results_df = results_dataframe(results)

    return (
        results_df,
        best_model,
        best_model_name,
        best_accuracy
    )