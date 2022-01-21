import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype #, is is_string_dtype, is_categorical_dtype

from collections import namedtuple

"""Functions and classes used by most other modules"""


class DataNum:

    """[Most classes use a pandas DataFrame providing numeric and possibly categorical order and data type.]
    """

    def __init__(self, data:pd.DataFrame, ncols: int = None) -> None:
        
        self._check_df(data)
        self.ncols = ncols if ncols else len(data.columns)
        self._check_dtypes(data, self.ncols)
        self.df = data
        self.var_names = self.df.columns.to_list()
        self.nans = (len(  self.df) - self.df.count()).sum()

    def __repr__(self) -> str:
        return (f"DataNum(Observations: {len(self.df)}, Features: {len(self.var_names)}, Total NaN: {self.nans})")
    
    def _check_df(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data needs to be supplied in pandas DataFrame, got: {type(data)}')

    def _check_dtypes(self, data, ncols):
        
        expected = np.repeat('floating', ncols)
        observed = data.apply(infer_dtype).values
        
        if len(expected) != len(observed):
            raise ValueError(f'Number of columns expected: {len(expected)}, got: {len(observed)}')
        
        if not all(expected == observed):
            raise TypeError(f"Expected {len(expected)} numeric columns, got {np.sum(observed=='floating')}")

            
        
 
