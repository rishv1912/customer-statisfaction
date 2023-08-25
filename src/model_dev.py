import logging
from abc import abstractmethod, ABC
from sklearn.linear_model import LinearRegression


class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the modle 
        Args: 
            X_train: Training data
            y_train :Training labels
        Returns :
            None
        """
        pass


class LinearRegressionModel(Model):
    """ Linear Regression Model"""

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args :
            X_train : Training data
            y_train : Training labels

        Returns :
            None        
        """
        try :
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Modle training completed")
            return reg
        except Exception as e :
            logging.error(f"Error in training model : {e}")
            raise e 