import logging
from abc import ABC, abstractmethod
import numpy as np

from sklearn.metrics import r2_score, mean_squared_error

class Evaluation(ABC):
    
    """
    Abstract class defining strategy for evalution for models
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        """
        Calculate the scores of the model
        
        Args:
        y_true (np.ndarray): True Labels
        y_pred (np.ndarray): Predicted Labels
        
        Returns None
        
        pass
        """
        
class MSE(Evaluation):
    
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        """
        Calculates the MSE
        
        Args:
        y_true : Actual labels
        y_pred: Predicted labels

        Returns:
            None
        """
        
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise(e)    
        
class R2(Evaluation):
    
    """
    Evaluation Strategy that uses R2
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        """
        Calculates the R2
        
        Args:
        y_true : Actual labels
        y_pred: Predicted labels

        Returns:
            None
    """
        try:
            logging.info("Calculate R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 Score: {}".format(e))
            raise e
        
        
class RMSE(Evaluation):
    
    """
     Evaluation strategy that uses RMSE
    """
    
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        
        """
        Calculates the RMSE
        
        Args:
        y_true : Actual labels
        y_pred: Predicted labels

        Returns:
            None
        """
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e