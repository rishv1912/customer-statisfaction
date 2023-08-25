import logging 
import os 
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error , r2_score

class Evaluation(ABC):
    """
    Abstract clas defiing stratedy for evaluation our models
    """

    @abstractmethod
    def calculate_scores(self,y_true : np.ndarray,y_pred:np.ndarray):
        """
        Calculates the scores for the model

        Args :
            y_true :True labels
            y_pred : Predicted labels
        Returns :
            None
        """

        pass


class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error"""

    def calculate_scores(self,y_true : np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true,y_pred)
            logging.info(f"MSE : {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE : {e}")
            raise e 
        
class R2(Evaluation):
    """ 
    Evaluation strategy that uses R2 Score"""


    def calculate_scores(self,y_true : np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating MSE")
            r2= r2_score(y_true,y_pred)
            logging.info(f"R2 Score : {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 Score : {e}")
            raise e 
        

class RMSE(Evaluation):
    def calculate_scores(self,y_true : np.ndarray,y_pred:np.ndarray):
        try:
            logging.info("Calculating MSE")
            rmse= mean_squared_error(y_true,y_pred)
            logging.info(f"RMSE : {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE : {e}")
            raise e 