from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier


def get_models():

    models = {

        "Logistic Regression":
        LogisticRegression(max_iter=1000),

        "Decision Tree":
        DecisionTreeClassifier(random_state=42),

        "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

        "KNN":
        KNeighborsClassifier(),

        "SVM":
        SVC(
            probability=True,
            random_state=42
        ),

        "XGBoost":
        XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    return models