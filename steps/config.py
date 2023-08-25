from zenml.steps import BaseParameters

class ModelNameConfig(BaseParameters):
    """Model Configs"""

    model_name :str = "LinearRegression"
    model_kwargs : dict = {}