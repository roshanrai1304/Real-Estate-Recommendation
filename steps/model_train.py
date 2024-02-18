import logging
import pandas as pd
from zenml import step
from sklearn.base import RegressorMixin



@step()
def train_model(
    df: pd.DataFrame
    # X_train: pd.DataFrame,
    # X_test: pd.DataFrame,
    # y_train: pd.Series,
    # y_test: pd.Series,
) -> None:
    
    """
    Trains the model on the ingested data.
    
    Args:
    df: the ingested data
    
    """
    pass