import numpy as np
import pandas as pd

"""Functions and classes used by most other modules"""



class DataNum:

    """[Most classes use a pandas DataFrame providing numeric and possibly categorical order and data type.]
    """

    def __init__(self, data:pd.DataFrame, ncols: int = None) -> None:
        
        self._check_df(data)
        self.ncols = ncols if ncols else len(data.columns)
        self._check_dtypes(data, self.ncols)
        self.data = data

    def _check_df(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data needs to be supplied in pandas DataFrame, got: {type(data)}')

    def _check_dtypes(self, data, ncols):
        try:
            expected = np.repeat('float', ncols)
            observed = np.array([data[col].dtype.name.rstrip('_12346') for col in data.columns])
            check = all(expected == observed)
        except ValueError:
            print(f'Number of data columns expected: {ncols}, got: {len(observed)}')
        else:
            if not check:
                raise ValueError(f"Expected {len(expected)} numeric columns, got {np.sum(observed=='float')}")