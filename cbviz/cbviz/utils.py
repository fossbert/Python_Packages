import numpy as np
import pandas as pd

# type checking
from pandas.api.types import infer_dtype, is_categorical_dtype, is_integer_dtype
from itertools import product
# as usual
from collections import namedtuple

"""Functions and classes used by most other modules"""


class DataNum:

    """[Most classes use a pandas DataFrame providing numeric and possibly categorical order and data type.]
    """

    def __init__(self, data:pd.DataFrame, ncols: int = None) -> None:
        
        self._check_df(data)
        self.ncols = ncols if ncols else len(data.columns)
        # integer columns get converted without telling anyone
        self._check_dtypes(data.copy().apply(int_cleaner), self.ncols) 
        self.df = data.copy()
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
            raise TypeError(f"Expected {len(expected)} float columns, got {np.sum(observed=='floating')}")



class DataMix(DataNum):

    """[This class prepares and checks DataFrames containing a numeric and one or two categorical variables]
    """

    def __init__(self, data:pd.DataFrame, ncat: int = 1, minsize:int = 5) -> None:
        
        super()._check_df(data)
        self.ncat = ncat
        self.dtypes = self._check_dtypes(data, self.ncat)
        self.df = data.copy().apply(cat_cleaner)
        self.var_names = self.df.columns.to_list()
        self.nans = (len(self.df) - self.df.count()).sum()
        self.minsize = self._check_minsize(minsize)

    def __repr__(self) -> str:
        
        return (f"DataNum(Obs total: {len(self.df)}, " 
                f"Features: {len(self.var_names)}, Total NaN: {self.nans}, "
                f"Obs in smallest subgroup: {self.minsize})")
 
    def _check_dtypes(self, data, ncat):
        
        expected_list = [(x, *y) for x in ['floating'] for y in list(product(['string', 'categorical'], repeat=ncat))]
        observed = data.apply(infer_dtype).values
        
        if ncat+1 != len(observed):
            raise ValueError(f'Number of columns expected: {ncat+1}, got: {len(observed)}')
        
        if not any([all(np.array(exp) == observed) for exp in expected_list]):
            raise TypeError(f"Could not verify data types, need: floating then {ncat} * string|categorical")

        return observed


    def _check_minsize(self, minsize: int):

        minsize_observed = self.df.groupby(self.var_names[1:]).count().values.ravel().min()
        if  minsize_observed < minsize:
            raise AssertionError(f"Need at least {minsize} observations per subgroup, found minimum of {minsize_observed }!")

        else:
            return minsize_observed        


def cat_cleaner(series):
    if is_categorical_dtype(series):
        return series.cat.remove_unused_categories()
    else:
        return series


def int_cleaner(series):
    if is_integer_dtype(series):
        return series.astype('float')
    else:
        return series
        


def _cut_p(pval):
    if pval < 0.001:
        return "***"
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return "*"
    else:
        return 'ns'