from unicodedata import name
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

# type checking
from pandas.api.types import infer_dtype, CategoricalDtype, is_integer_dtype
from itertools import product
# as usual
from collections import namedtuple

"""Functions and classes used by most other modules"""


class DataNum:

    """[Most classes use a pandas DataFrame providing numeric and possibly categorical order and data type.]
    """

    def __init__(self, data:pd.DataFrame, ncols: int = None) -> None:
        
        self._check_df(data)
        data = data.apply(series_cleaner)
        self.ncols = ncols if ncols else len(data.columns)
        # integer columns get converted without telling anyone
        self._check_dtypes(data, self.ncols) 
        self.df = data.copy()
        self.var_names = self.df.columns.to_list()
        self.nans = (len(self.df) - self.df.count()).sum()

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
        data = data.apply(series_cleaner)
        self.ncat = ncat
        self.dtypes = self._check_dtypes(data, self.ncat)
        self.df = data.copy()
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


class DataDot(DataNum):

    def __init__(self, data: pd.DataFrame, x:str, y:str, size:str, color:str=None) -> None:

        super()._check_df(data)
        data = data.apply(series_cleaner)
        self.var_names = Vars(x, y, size, color)
        self.n_numeric = 2 if color else 1
        self._check_var_names(self.var_names, data)
        self.dtypes = self._check_dtypes(data, self.n_numeric)
        self.df = data[[var for var in self.var_names if var]].copy().apply(series_cleaner) # Messy! 
        self.ncols, self.nrows, *_ = self.df.nunique()

    def __repr__(self) -> str:
        return (f"DataDot(Obs total: {len(self.df)}, " 
                f"X and Y variables: {', '.join([self.var_names.x, self.var_names.y])}. "
                f"Shape: {self.nrows} rows, {self.ncols} columns, "
                f"Number of annotating features: {self.n_numeric})")        

    def _check_var_names(self, vars:tuple, data: pd.DataFrame):
        
        if not all([var in data.columns for var in vars if var]):
            raise KeyError(f'Could not find all provided keys ({", ".join(vars)}) in DataFrame')
        

    def _check_dtypes(self, data, n_numeric):

        expected_list = [(*x, *y) for x in  list(product(['string', 'categorical'], repeat=2)) for y in list(product(['integer', 'floating'], repeat=n_numeric))]
        observed = data.apply(infer_dtype).values
        
        if n_numeric+2 != len(observed):
            raise ValueError(f'Number of columns expected: {n_numeric+2}, got: {len(observed)}')
        
        if not any([all(np.array(exp) == observed) for exp in expected_list]):
            raise TypeError(f"Could not verify data types, need: 2*string|categorical, then {n_numeric}*integer|floating")

        return observed

      





### Helper functions from here on

Vars = namedtuple('Vars', 'x y size color')

def series_cleaner(series):
    
    if isinstance(series.dtype, CategoricalDtype):
        return series.cat.remove_unused_categories()
    
    elif is_integer_dtype(series):
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
    elif pval < 0.1:
        return f'{pval:.2f}'
    else:
        return 'ns'
    
def _color_light_or_dark(rgba_in:np.ndarray)-> str:
    """[For plotting purposes, we determine whether a color is light or dark and adjust its text color accordingly.
    Also see https://stackoverflow.com/questions/22603510/is-this-possible-to-detect-a-colour-is-a-light-or-dark-colour]

    Args:
        rgba_in ([np.ndarray]): [A numpy array containing RGBA as returned by matplotlib colormaps]

    Returns:
        [str]: [A string: w for white or k for black]
    """
    r,g,b,_ = rgba_in*255
    hsp = np.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
    if (hsp>127.5):
        # light color, return black for text
        return 'k'
    else:
        # dark color, return white for text
        return 'w'
    

    
def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    """
    Also see https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10
    """
    
    if nc > plt.get_cmap(cmap).N:
        raise ValueError(f"Too many categories for colormap: {cmap}.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0, 1, nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv, nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1], 0.25, nsc)
        arhsv[:,2] = np.linspace(chsv[2], 1, nsc)
        rgb = colors.hsv_to_rgb(arhsv[::-1,:]) #modification to have ascending hues left to right
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    #cmap = colors.ListedColormap(cols)
    return cols