from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configs"""
    model_name: str = "Stacking"
    
    """Stacking weights"""
    lasso_weight = 0.047
    ridge_weight = 0.2
    svr_weight = 0.25
    ker_weight = 0.3
    elastic_weight = 0.003
    bay_weight = 0.2