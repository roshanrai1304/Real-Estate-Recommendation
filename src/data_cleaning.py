import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from pandas.core.api import Series as Series

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

"""
    The purpose of abstract methods is to define a method in an abstract base class 
    without providing an implementation. Subclasses are then required to provide their 
    own implementation of the abstract method.
"""

#abstract class
class DataStrategy(ABC): 
    
    """
    Abstract class defining strategy for handling data
    """
    
    """
    The type hint Union[pd.DataFrame, pd.Series] is specifying that a function or 
    method should return either a Pandas DataFrame or a Pandas Series.
    """
    
    @abstractmethod
    def handle_data(self, data: Union[pd.DataFrame, Series]) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        pass
    

class DataPreProcessingStrategy(DataStrategy):
    """
    Startegy for processing data

    """
    
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame :
        """
        Preprocess data

        Args:
            data (pd.DataFrame)

        Returns:
            pd.DataFrame
            
        """
        try:
            data = data.drop(
                ["builderName", "State", "District", "flatNumber", "yogaArea", "Garden", "playArea", "Parking", "Gym", "Location", "Highway", "Railway", "Metro", "busStop",
                 "closestHospitalDistance", "closestMallDistance", "closestBusinessParkDistance", "noOfBathrooms", "swimmingPool", "ATM/Finance"],
                axis=1
            )
            data = data[data['flatType'] != 'REFUGE']
            data = data[data['Area'] != 'REFUGE']
            data = data[data['pricePerSqFeet'] != 'REFUGE']
            data = data[data['totalPrice'] != 'REFUGE']
            
            object_columns = []
            for i in range(0, len(data.columns)):
                if data.iloc[:,i].dtype == 'object':
                    object_columns.append(data.iloc[:,i].name)
                    
            for i in object_columns:
                data[i] = pd.to_numeric(data[i])
                
                
            data['totalPrice'] = np.log(data['totalPrice'])
            data['Area'] = np.log(data['Area'])
            data['pricePerSqFeet'] = np.log(data['pricePerSqFeet'])
            print(data)
            data_columns = data.columns
            # data_transform = scaling.transform(data.to_numpy())
            # print(data_scaled)    
            data = pd.DataFrame(data, columns=data_columns)            
             
            return data
        
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e
            

class DataDivideStrategy(DataStrategy):
    
    """
    Strategy for dividing data into train and test
    """
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test set

        Args:
            data (pd.DataFrame)

        Returns:
            Union[pd.DataFrame, pd.Series]
        """
        
        try:
            X = data.drop(['totalPrice'], axis=1)
            y = data['totalPrice']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            scaling_X = StandardScaler().fit(X_train)
            scaling_y = StandardScaler().fit(y_train.to_numpy().reshape(-1,1))
            X_train = scaling_X.transform(X_train.to_numpy())
            X_test = scaling_X.transform(X_test.to_numpy())
            y_train = scaling_y.transform(y_train.to_numpy().reshape(-1,1))
            print(f"y_train -> {y_train.shape}")
            y_test = scaling_y.transform(y_test.to_numpy().reshape(-1,1))
            with open("scalers/scaling_X.pkl", 'wb') as f:
              pickle.dump(scaling_X, f)
            with open("scalers/scaling_y.pkl", 'wb') as f:
              pickle.dump(scaling_y, f)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data {}".format(e))
            raise e
     
        
class DataCleaning:
    
    """
    Class for cleaning data which processes the data and divides it into train and test
    """
    
    def __init__(self, data:pd.DataFrame, strategy: DataStrategy):
        self.data =  data
        self.strategy = strategy
        
    def handle_data(self) -> Union[pd.DataFrame, pd.Series, np.ndarray]:
        """
        handle data

        Returns:
            Union[pd.DataFrame, pd.Series, np.ndarray]
        """
        
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e
        
if __name__ == "__main__":
    
    data = pd.read_csv(f"data/{os.listdir('data')[0]}")
    columns_for_df = ['closestEducationalInstituteDistance',
                'flatType',
                'Area',
                'pricePerSqFeet',
                'shoppingArea']
    # data = data[columns_for_df].iloc[0:10, :]
    data = data.iloc[:, 1:]
    data = data.iloc[:10,:]
    # print(data)
    data_cleaning = DataCleaning(data.iloc[:10,:], DataPreProcessingStrategy())
    data = data_cleaning.handle_data()
    data_divide = DataCleaning(data, DataDivideStrategy())
    X_train, X_test, y_train, y_test, scaling_X, scaling_y = data_divide.handle_data()
    print((X_test))
    print(scaling_X.inverse_transform(X_test))
    print("yes")
    X_test
    
            