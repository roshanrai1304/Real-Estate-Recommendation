import logging
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import cross_val_score
from sklearn.base import RegressorMixin

from src.model_dev import LinearRegressionModel
from src.model_dev import LassoModel
from src.model_dev import RidgeModel
from src.model_dev import RandomForestRegressorModel
from src.model_dev import GradientBoostingRegressorModel
from src.model_dev import SVRModel
from src.model_dev import LinearSVRModel
from src.model_dev import ElasticNetModel
from src.model_dev import KernelRidgeModel
from src.model_dev import BayesianRidgeModel
from src.model_dev import Stacking



def rmse_cv(model, features, label):
    rmse = np.sqrt(-cross_val_score(model, features, label, scoring="neg_mean_squared_error",cv=5))
    return rmse

def train_model(
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> RegressorMixin:
    
    """
    Trains the model on the ingested data.
    
    Args:
    df: the ingested data
    
    """
    try:
        model= model_name
        # model = config.model_name
        
        if model == "Stacking":
            lasso = LassoModel(alpha= 0.0005, max_iter= 10000)
            ridge = RidgeModel(alpha=45, max_iter= 10000)
            svr = SVRModel(C = 0.2, epsilon= 0.025, gamma = 0.0004, kernel = 'rbf')
            ker = KernelRidgeModel(alpha=0.15 ,kernel='polynomial',degree=3 , coef0=0.9)
            elastic_net = ElasticNetModel(alpha=0.0065,l1_ratio=0.075,max_iter=10000)
            bayesian = BayesianRidgeModel()
            
            stack_model = Stacking(mod=[lasso, ridge, svr, ker, elastic_net, bayesian], meta_model=ker)
            print(f"The type of stack_model is {type(stack_model)}")
            # score = rmse_cv(stack_model, X_train, y_train)
            print(f"y_train -> {y_train.shape}")
            stack_model.fit(X_train, y_train)
            # print(f"The score is -> {score.mean()}")
            return stack_model
        
        elif model == "LinearRegression":
            linear = LinearRegressionModel()
            linear.fit(X_train, y_train.ravel())
            return linear
        
        elif model == "RandomForest":
            random_forest = RandomForestRegressorModel()
            random_forest.fit(X_train, y_train)
            return random_forest
        
        elif model == "GradientBoost":
            gradient = GradientBoostingRegressorModel()
            gradient.fit(X_train, y_train)
            return gradient
            
        elif model == "SVR":
            svr = SVRModel()
            svr.fit(X_train, y_train)
            return svr
        else:
            raise ValueError("Model {} not supported".format(model_name))
        
    except Exception as e:
        logging.error("Error in training model: {}, {}".format(model_name, e))
        raise e