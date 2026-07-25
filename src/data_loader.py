import os
import pandas as pd


def load_data():
    """
    Load Customer Churn dataset.
    """

    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    data_path = os.path.join(
        base_dir,
        "data",
        r"C:\Users\Linga Raju\Documents\churn_predictionpro\Churn_Modelling.csv"
    )

    df = pd.read_csv(data_path)

    return df