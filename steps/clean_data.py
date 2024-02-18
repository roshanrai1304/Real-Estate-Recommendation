import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated



@step
def clean_df(df: pd.DataFrame) -> None:
    pass