import mlflow
import logging
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from urllib.parse import urlparse
from steps.config import model_names
from steps.predict_results import compare_results
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd


# tracking_url_type_store = None
tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
#
def train_pipelines(data_path: str):
    df = ingest_df(data_path)
    df = df.iloc[:,1:]
    X_train, X_test, y_train, y_test, scaling_X, scaling_y = clean_df(df)
    print(f"X_test.shape -> {X_test.shape}")
    for model_type in model_names:
        with mlflow.start_run():
            try:
                model = train_model(model_type, X_train, X_test, y_train, y_test)
                r2_score, rmse, y_pred = evaluate_model(model, X_test, y_test)
                data = compare_results(X_test, y_test, y_pred, scaling_X, scaling_y)
                data.to_csv(f"predicted_results_{model_type}.csv")
                if tracking_url_type_store != "file":
                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(
                        model, "model", registered_model_name="Model"
                    )
                    logging.info(f"{model_type} model evalution is completed")
                else:
                    mlflow.sklearn.log_model(model, "model")
            except Exception as e:
                logging.info(f"Error in training {model_type}")
        
    