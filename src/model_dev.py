import logging
import joblib
from abc import ABC, abstractmethod
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin,clone
import numpy as np



class Model(ABC, BaseEstimator, RegressorMixin):
    """
    Abstract class for all models

    """
    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Train the model

        Args:
            X_train : Traininig data
            y_train : Traininig label
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Makes predictions using the trained model

        Args:
            X (type): Input data for prediction
            
        Returns:
            type: Predictions
        """
        pass
    
class StackedModel(ABC, BaseEstimator, TransformerMixin, RegressorMixin):
    """
    Abstract class for stacked models

    """
    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Train the model

        Args:
            X_train : Traininig data
            y_train : Traininig label
        """
        pass
    
    
    @abstractmethod
    def predict(self, X_train):
        """
        Generates predictions for new data using the trained stacking ensemble.
        Combines predictions from each base model for each fold and calculates the mean.
        Uses the mean predictions as features for the meta-model to make the final prediction.

        Args:
            X (type): Input data for prediction
            
        Returns:
            type: Predictions
        """
        pass
    
    @abstractmethod
    def get_oof(self, X, ):
        """
        Generates out-of-fold predictions for the training data and mean predictions for the test data.
        Similar to the fit method but collects out-of-fold predictions for both training and test data separately.

        Args:
            X (type): Input data for prediction
            
        Returns:
            type: Predictions
        """
        pass
    
    @abstractmethod
    def save_model(self, filename):
        """
        Save the stacking ensemble model to a file.

        Parameters:
        - filename (str): The name of the file to save the model.
        """
        pass
    
    @abstractmethod
    def load_model(self, filename):
        """
        Load the stacking ensemble model to a file.

        Parameters:
        - filename (str): The name of the file to save the model.
        """
        pass
    


    
class LinearRegressionModel(Model):
    
    """
    Linear Regression Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = LinearRegression(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model LinearRegression: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
    
        
class LassoModel(Model):
    
    """
    Lasso Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = Lasso(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model Lasso: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
    
        
class RidgeModel(Model):
    
    """
    Ridge Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = Ridge(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model Ridge: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
    
        
class RandomForestRegressorModel(Model):
    
    """
    Random Forest Regressor Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model RandomForestRegressor: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
    
        
class GradientBoostingRegressorModel(Model):
    
    """
    GradientBoostingRegressor Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = GradientBoostingRegressor(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model GradientBoostingRegressor: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
    
        
class SVRModel(Model):
    
    """
    SVR Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = SVR(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model SVR: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
    
        

class LinearSVRModel(Model):
    
    """
    LinearSVR Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = LinearSVR(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model LinearSVR: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
    
        

class ElasticNetModel(Model):
    
    """
    ElasticNet Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = ElasticNet(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model ElasticNet: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
    
        

class KernelRidgeModel(Model):
    
    """
    KernelRidge Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = KernelRidge(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model KernelRidge: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
    
        

class BayesianRidgeModel(Model):
    
    """
    BayesianRidge Model
    
    Args:
      X_train: Training features
      y_train: Training labels
    """
    
    def __init__(self, **kwargs):
        self.model = BayesianRidge(**kwargs)
        
        
    def fit(self, X_train, y_train):
        """
        Trains the model

        Args:
            X_train : Training features
            y_train : Training labels
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed")
            return self
        except Exception as e:
            logging.error("Error in training model BayesianRidge: {}".format(e))
            raise e
        
    def predict(self, X_test):
        """
        predicts for test_data

        Args:
            X_test
        """
        return self.model.predict(X_test)
    
    
class Stacking(StackedModel):
    def __init__(self,mod,meta_model, **kwargs):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(**kwargs)
        
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))
        
        for i,model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X,y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index,i] = renew_model.predict(X[val_index])
        
        self.meta_model.fit(oof_train,y)
        return self
    
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) for single_model in self.saved_model]) 
        return self.meta_model.predict(whole_test)
    
    def get_oof(self,X,y,test_X):
        oof = np.zeros((X.shape[0],len(self.mod)))
        test_single = np.zeros((test_X.shape[0],5))
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index],y[train_index])
                oof[val_index,i] = clone_model.predict(X[val_index])
                test_single[:,j] = clone_model.predict(test_X)
            test_mean[:,i] = test_single.mean(axis=1)
        return oof, test_mean
    
    def save_model(self, filename):
        """
        Save the stacking ensemble model to a file.

        Parameters:
        - filename (str): The name of the file to save the model.
        """
        joblib.dump(self, filename)
        
    @classmethod
    def load_model(cls, filename):
        """
        Load the stacking ensemble model from a file.

        Parameters:
        - filename (str): The name of the file containing the saved model.

        Returns:
        - Stacking: The loaded stacking ensemble model.
        """
        return joblib.load(filename)
    
       
    

    
        

