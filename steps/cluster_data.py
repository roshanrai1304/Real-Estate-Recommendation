import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
import pickle

from src.data_cleaning import DataCleaning
from src.data_cleaning import DataPreProcessingStrategy

from sklearn.cluster import KMeans




def handle_data(data: pd.DataFrame) -> None:
    
    preprocess_strategy = DataPreProcessingStrategy()
    data_cleaning = DataCleaning(data, preprocess_strategy)
    preprocessed_data = data_cleaning.handle_data()
    preprocessed_data = pd.DataFrame(preprocessed_data)
    with open("scalers/scaling_X.pkl", 'rb') as f:
            sc_X = pickle.load(f)
    df_scaled = sc_X.transform(preprocessed_data.drop(['totalPrice'], axis=1))
    kmeans = KMeans(n_clusters=20, random_state=42)
    data['cluster'] = kmeans.fit_predict(df_scaled)
    with open("clusters/kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    data.to_csv("clusters/cluster_data.csv")
    return None