import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pickle

#abstract class
class InverseDataStrategy(ABC):
    
    """
    Abstract class for Inverse processing data
    """
    
    @abstractmethod
    def inverse_data(self, X_test: np.ndarray, y_true:np.ndarray, y_predict:np.ndarray, scaler_X:StandardScaler, scaler_y:StandardScaler) -> pd.DataFrame:
        pass
    
class InverseDataProcessing(InverseDataStrategy):
    
    """
    Strategy for inverse processing data
    """
    
    def inverse_data(self, X_test: np.ndarray, y_true:np.ndarray, y_pred:np.ndarray) -> pd.DataFrame:
        
        """
        Inverse process the data to its original values
        
        Args:
         data: np.ndarray
        
        Returns:
             
        """
        try:
            columns_for_df = ['closestEducationalInstituteDistance',
                    'flatType',
                    'Area',
                    'pricePerSqFeet',
                    'shoppingArea']
            with open("scalers/scaling_X.pkl", 'rb') as f:
                scaler_X = pickle.load(f)
            with open("scalers/scaling_y.pkl", 'rb') as g:
                scaler_y = pickle.load(g)
            
            X_test = scaler_X.inverse_transform(X_test)
            y_true = np.exp(scaler_y.inverse_transform(y_true.reshape(-1,1)))
            y_pred = np.exp(scaler_y.inverse_transform(y_pred.reshape(-1,1)))
            X_test = pd.DataFrame(X_test, columns=columns_for_df)
            X_test['Area'] = np.exp(X_test['Area'])
            X_test['pricePerSqFeet'] = np.exp(X_test['pricePerSqFeet'])
            # print(f"inverse data, {y_true}")
            dic = {
                    "totalSalePrice": y_true.reshape(1, -1)[0],
                    "predictedSalePrice": y_pred.reshape(1, -1)[0]
                }
            y_results = pd.DataFrame(dic)
            data = pd.concat([X_test, y_results], axis=1)
            return data
        except Exception as e:
            logging.error("Error in inverse processing the data {}".format(e))
            

        
        
        