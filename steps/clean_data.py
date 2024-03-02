import logging
import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
from pandas.core.api import Series
from typing import Union
from src.data_cleaning import DataCleaning
from src.data_cleaning import DataPreProcessingStrategy
from src.data_cleaning import DataDivideStrategy



def clean_df(df: Union[pd.DataFrame, Series]) ->Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleans the data and divde it into train and test also prepares the data for stacking
    
    Args:
      df: Raw data
      
    Returns:
    X_train: Training data
    X_test: Testing data
    y_train: Training labels
    y_test: Testing labels
    
    """
    
    try:
      preprocess_strategy = DataPreProcessingStrategy()
      data_cleaning = DataCleaning(df, preprocess_strategy)
      preprocessed_data = data_cleaning.handle_data()
      
      divide_strategy = DataDivideStrategy()
      data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
      X_train, X_test, y_train, y_test = data_cleaning.handle_data()
      logging.info("Data cleaning completed")
      return X_train, X_test, y_train, y_test
    except Exception as e:
      logging.error("Error in cleaning data: {}".format(e))
      raise e  
    
    