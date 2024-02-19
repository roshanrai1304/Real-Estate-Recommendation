import logging
import mlflow
import pandas as pd
import numpy as np

from sklearn.base import RegressorMixin
from src.evaluation import MSE, R2, RMSE

from typing import Tuple
from typing_extensions import Annotated

from zenml import step
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
                   X_test: np.ndarray,
                   y_test: np.ndarray
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]] :
    
    """
    Args:
     model: RegressionMixin
     X_test: pd.DataFrame
     y_test: pd.Series
     
    Return:
     r2_score: float
     rmse: float
    """
    
    try:
      prediction = model.predict(X_test)
      mse_class = MSE()
      mse = mse_class.calculate_scores(y_test, prediction)
      mlflow.log_metric("mse", mse)
      
      r2_class = R2()
      r2 = r2_class.calculate_scores(y_test, prediction)
      mlflow.log_metric("r2", r2)

      rmse_class = RMSE()
      rmse = rmse_class.calculate_scores(y_test, prediction)
      mlflow.log_metric("rmse", rmse)

      return r2, rmse
    except Exception as e:
      logging.error("Error in evaluating model: {}".format(e))
      raise e
    