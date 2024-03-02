import logging
import pandas as pd
import numpy as np

from typing_extensions import Annotated
from src.inverse_preprocess_data import InverseDataProcessing
from sklearn.preprocessing import StandardScaler

def compare_results( X_test: np.ndarray,
                    y_true: np.ndarray,
                    y_pred: np.ndarray,
                    scaler_X:StandardScaler,
                    scaler_y:StandardScaler
) -> Annotated[pd.DataFrame, "predicted_results"]:
    
    """
    Args:
      X_test: numpy array
      y_true: numpy array
      y_pred: numpy array
    """
    try:
        inverse_data = InverseDataProcessing()
        data = inverse_data.inverse_data(X_test, y_true, y_pred, scaler_X, scaler_y)
        logging.info("Inverse Data is completed")
        return data
    except Exception as e:
        logging.error("Error in Inverse transform data".format(e))
        raise e
       