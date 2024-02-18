import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from pandas.core.api import Series as Series
from sklearn.model_selection import train_test_split

"""
    The purpose of abstract methods is to define a method in an abstract base class 
    without providing an implementation. Subclasses are then required to provide their 
    own implementation of the abstract method.
"""

#abstract class
class DataStrategy(ABC): 
    
    """
    Abstract class defining strategy for handling data
    """
    
    """
    The type hint Union[pd.DataFrame, pd.Series] is specifying that a function or 
    method should return either a Pandas DataFrame or a Pandas Series.
    """
    
    @abstractmethod
    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    

class DataPreProcessingStrategy(DataStrategy):
    """
    Startegy for processing data

    """
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame :
        """
        Preprocess data

        Args:
            data (pd.DataFrame)

        Returns:
            pd.DataFrame
            
        """
        try:
            data = data.drop(
                ["builderName", "State", "District", "flatNumber", "yogaArea", "Garden", "playArea", "Parking", "Gym"],
                axis=1
            )
            
            data = data[data['flatType'] != 'REFUGE']
            data = data[data['Area'] != 'REFUGE']
            data = data[data['noOfBathrooms'] != 'REFUGE']
            data = data[data['pricePerSqFeet'] != 'REFUGE']
            data = data[data['totalPrice'] != 'REFUGE']
            
            object_columns = []
            for i in range(0, len(data.columns)):
                if data.iloc[:,i].dtype == 'object':
                    object_columns.append(data.iloc[:,i].name)
                    print(data.iloc[:,i].dtype)
                    
            for i in object_columns:
                data[i] = pd.to_numeric(data[i])
             
            return data
        
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            


class DataDivideStrategy(DataStrategy):
    
    """
    Strategy for dividing data into train and test
    """
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test set

        Args:
            data (pd.DataFrame)

        Returns:
            Union[pd.DataFrame, pd.Series]
        """
        
        try:
            features = ['Area', 'pricePerSqFeet', 'closestEducationalInstituteDistance', 'totalPrice', 'flatType', 'shoppingArea']
            target = ['totalPrice']
            X = data[features]
            y = data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data {}".format(e))
            raise e
        
        

        
        
class DataCleaning:
    
    """
    Class for cleaning data which processes the data and divides it into train and test
    """
    
    
    
    
            