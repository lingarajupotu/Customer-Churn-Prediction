from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from config import RANDOM_STATE, CV_FOLDS


def tune_random_forest(preprocessor, X_train, y_train):
    """
    Tune Random Forest using GridSearchCV.
    """

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE))
        ]
    )

    param_grid = {

        "model__n_estimators": [100, 200, 300],

        "model__max_depth": [5, 10, 15],

        "model__min_samples_split": [2, 5, 10]
    }

    grid_search = GridSearchCV(

        estimator=pipeline,

        param_grid=param_grid,

        cv=CV_FOLDS,

        scoring="accuracy",

        n_jobs=-1,

        verbose=2
    )

    grid_search.fit(X_train, y_train)

    return grid_search