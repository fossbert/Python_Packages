from unicodedata import name
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib imports
from matplotlib import colors
from matplotlib.scale import FuncScale

# type checking
from pandas.api.types import infer_dtype, CategoricalDtype, is_integer_dtype
from itertools import product
# as usual
from collections import namedtuple


# survival data functionality
from sksurv.util import Surv

"""Functions and classes used by most other modules"""


class DataNum:

    """[Most classes use a pandas DataFrame providing numeric and possibly categorical order and data type.]
    """

    def __init__(self, data:pd.DataFrame, ncols: int = None) -> None:
        
        self.nans_removed = 0
        self._check_df(data)
        data = data.apply(series_cleaner)
        self.ncols = ncols if ncols else len(data.columns)
        # integer columns get converted without telling anyone
        self._check_dtypes(data, *['floating']*self.ncols) 
        self.df = self._check_for_nan(data.copy())
        self.var_names = self.df.columns.to_list()

    def __repr__(self) -> str:
        return (f"DataNum(Observations: {len(self.df)}, Features: {len(self.var_names)}, NaN removed: {self.nans_removed})")
    
    def _check_df(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f'Data needs to be supplied in pandas DataFrame, got: {type(data)}')

    def _check_dtypes(self, data, *dtypes, return_dtypes:bool=False):
        
        allowed = _generate_dtype_options(*dtypes)
        observed = data.apply(infer_dtype).values

        if not any(len(arr)==len(observed) for arr in allowed):
            raise ValueError(f"Number of supplied columns does not match expections of {', '.join(dtypes)}")
        
        if not any([all(arr == observed) for arr in allowed]):
            raise TypeError(f"Could not verify data types for this class, need {', '.join(dtypes)}")
        
        if return_dtypes:
            return observed
        
    def _check_for_nan(self, data):

        nans = (len(data) - data.count()).sum()
        if nans>0:
            self.nans_removed = nans
            print(f"Found {nans} missing values, removing them.")
            return data.dropna(axis=0)
        else:
            return data



class DataMix(DataNum):

    """[This class prepares and checks DataFrames containing a numeric and one or two categorical variables]
    """

    def __init__(self, data:pd.DataFrame, ncat: int = 1, minsize:int = 5) -> None:
        
        super()._check_df(data)
        self.nans_removed = 0
        data = data.apply(series_cleaner)
        self.ncat = ncat
        self.dtypes = super()._check_dtypes(data, 'floating', *['string|categorical']*self.ncat, return_dtypes=True)
        self.df = super()._check_for_nan(data.copy())
        self.var_names = self.df.columns.to_list()
        self.minsize = self._check_minsize(minsize)

    def __repr__(self) -> str:
        
        return (f"DataNum(Obs total: {len(self.df)}, " 
                f"Features: {len(self.var_names)}, NaN removed: {self.nans_removed}, "
                f"Obs in smallest subgroup: {self.minsize})")
 
    def _check_minsize(self, minsize: int):

        minsize_observed = self.df.groupby(self.var_names[1:], observed=True).count().values.ravel().min()
        if  minsize_observed < minsize:
            raise AssertionError(f"Need at least {minsize} observations per subgroup, found minimum of {minsize_observed }!")

        else:
            return minsize_observed        


class DataDot(DataNum):

    def __init__(self, data: pd.DataFrame, x:str, y:str, size:str, color:str=None) -> None:

        super()._check_df(data)
        self.nans_removed = 0
        data = data.apply(series_cleaner)
        self.var_names = Vars(x, y, size, color)
        self.n_numeric = 2 if color else 1
        self._check_var_names(self.var_names, data)
        self.dtypes = super()._check_dtypes(data, "string|categorical", "string|categorical", *['floating']*self.n_numeric, return_dtypes=True)
        self.df = super()._check_for_nan(data[[var for var in self.var_names if var]].copy()) # Messy! 
        self.ncols, self.nrows, *_ = self.df.nunique()

    def __repr__(self) -> str:
        return (f"DataDot(Obs total: {len(self.df)}, " 
                f"X and Y variables: {', '.join([self.var_names.x, self.var_names.y])}. "
                f"Shape: {self.nrows} rows, {self.ncols} columns, "
                f"Number of annotating features: {self.n_numeric})")        

    def _check_var_names(self, vars:tuple, data: pd.DataFrame):
        
        if not all([var in data.columns for var in vars if var]):
            raise KeyError(f'Could not find all provided keys ({", ".join(vars)}) in DataFrame')


class DataSurv(DataNum):

    def __init__(self, data: pd.DataFrame, ncat: int = 1) -> None:

        super()._check_df(data)
        self.nans_removed = 0
        data = data.apply(series_cleaner)
        self.ncat = ncat
        self.dtypes = super()._check_dtypes(data, 'floating', 'floating', *['string|categorical']*self.ncat, return_dtypes=True)
        self.df = super()._check_for_nan(data.copy())
        self.time, self.event, *self.groups = self.df.columns.to_list()
        self._check_event(self.df[self.event])
        self.surv = Surv.from_dataframe(self.event, self.time, self.df)

    def __repr__(self) -> str:
        return (
                f"DataSurv(Obs total: {len(self.df)}, NaN removed: {self.nans_removed},\n" 
                f"Time column: {self.time}, event column: {self.event},\n"
                f"Annotating feature(s): {' '.join([f'{i}. {g}' for i, g in enumerate(self.groups, 1)])})"
                )
    
    def _check_event(self, s:pd.Series):

        if not s.isin([0, 1]).all() and not (s == 0).all() and not (s == 1).all():

            raise ValueError("Event column not suitable, please double check!")



#################################################################
############## Helper functions from here on ####################
#################################################################

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


def _generate_dtype_options(*dtypes):
    
    """Helper function that generates a list of acceptable data types (also lists) based on a variable number and order of input data types"""

    out = []

    for i, dt in enumerate(dtypes):

        if i==0:
            if '|' in dt:
                    for opt in [['string'], ['categorical']]:
                        out.append(opt)
            else:
                out.append([dt])
        else:
            cp_list = out[:] # Create a copy of list

            for e in cp_list:
                if '|' in dt:
                    for opt in [['string'], ['categorical']]:
                        out.append(e + opt)
                    out.remove(e)
                else:                        
                    out.append(e + [dt])
                    out.remove(e)

    out = [np.array(e) for e in out] # convert to arrays

    return out