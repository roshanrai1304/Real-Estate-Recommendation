import logging
import pandas as pd
from src.data_cleaning import DataCleaning, DataPreProcessingStrategy, DataDivideStrategy

def get_data_for_test():
    df = pd.read_csv("data/processed_real_estate_data -  newLabel.csv")
    df = df.sample(n=1000)
    df = df.iloc[:, 1:]
    preprocess_strategy = DataPreProcessingStrategy()
    data_cleaning = DataCleaning(df, preprocess_strategy)
    preprocessed_data = data_cleaning.handle_data()
    
    col_names = list(preprocessed_data.columns)
    col_names.remove("totalPrice")
    
    divide_strategy = DataDivideStrategy()
    data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
    X_train, X_test, y_train, y_test = data_cleaning.handle_data() 
    return pd.DataFrame(X_test, columns=col_names).to_json(orient="split")