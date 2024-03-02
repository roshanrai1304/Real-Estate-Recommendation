import logging

import pandas as pd

class IngestData:
    
    """
    Ingesting the data from data_path
    """
    
    def __init__(self, data_path) :
        """
        
        Args:
            data_path: path to the data
        """
        self.data_path = data_path
        
    def get_data(self):
        """
        Ingesting the data from data_path
        
        Returns:
        dataframe: datafile
        
        """
        logging.info(f"Ingesting the data from {self.data_path}")
        return pd.read_csv(self.data_path)
    

def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from data_path

    Args:
        data_path (str): path to the data

    Returns:
        pd.DataFrame: the ingested data
    """
    
    try:
        ingest_df = IngestData(data_path)
        df = ingest_df.get_data()
        return df
    except Exception as e:
        logging.error("Error while ingesting data: {e}")
        raise e