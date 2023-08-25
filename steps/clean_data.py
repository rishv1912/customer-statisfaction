import logging
from zenml import step
import pandas as pd
from src.data_cleaning import DataCleaning, DataDivideStartegy, DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple


@step
def clean_df(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]: 
    ''' 
    cleans the data and divides it into train and test
    Args:
        df:Raw data
    Returns:
        X_train : Training data    
        X_test : Testing data
        y_train : Training labels
        y_test : Testing labels 
    '''



    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(data, process_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStartegy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        return  X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(e)
        raise e
