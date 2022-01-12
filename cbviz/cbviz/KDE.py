# import
import numpy as np
import pandas as pd
from itertools import product
from scipy.stats import gaussian_kde


"""Functions for univariate distributions"""

class SplitViolin:
    
    """Container for producing a split violin plot and compute associated statistics"""

    def __init__(self, data:pd.DataFrame, s1_order: list = None, s2_order: list = None, scale_factor:int = 2) -> None:
        
        self.dtype_options = [('float', *i) for i in product(['object', 'category'], repeat=2)]
        self.scale_factor = scale_factor
        
        
        # First round of checks tests input DataFrame
        # it's this complicated because the order of variables matters
        # i.e. float goes first, then two object|category variables
        
        if isinstance(data, pd.DataFrame):            
            if any([all(data.dtypes == list(dt_opt)) for dt_opt in self.dtype_options]):
                    self.ylabel, self.s1, self.s2 = data.columns
                    _, self.s1_dtype, self.s2_dtype = data.dtypes
                    self.data_in = data
            else:    
                raise ValueError(f"DataFrame needs three columns, dtypes allowed are: float - object|category - object|category")
        else:
            raise ValueError('Need a pandas DataFrame')
        
        # Second checks the split variables: 
        # 1.) Do they need conversion ? 
        # 2.) s2 can only have two levels 
        # 3.) If order is provided, reorder if provided levels match
        if self.s1_dtype == 'object':
            self.data_in[self.s1] = self.data_in[self.s1].astype('category')
        if self.s2_dtype == 'object':
            self.data_in[self.s2] = self.data_in[self.s2].astype('category')
        
        self.s1_categories = self.data_in[self.s1].cat.categories.to_list()
        self.s2_categories = self.data_in[self.s2].cat.categories.to_list()
        
        if s2_ncat := len(self.s2_categories) != 2:
            raise ValueError(f'Need 2 levels for {self.s2}, found {s2_ncat}') 
        
        if s1_order:
             if all([s1_lvl in self.s1_categories for s1_lvl in s1_order]):
                 self.data_in[self.s1] = self.data_in[self.s1].cat.reorder_categories(s1_order)
                 self.s1_categories = s1_order
             else:
                 raise ValueError(f'Could not align levels, provided: {", ".join(s1_order)}, available: {", ".join(self.s1_categories)}')
        
        if s2_order:
            if all([s2_lvl in self.s2_categories for s2_lvl in s2_order]):
                self.data_in[self.s2] = self.data_in[self.s2].cat.reorder_categories(s2_order)
                self.s2_categories = s2_order
            else:
                raise ValueError(f'Could not align levels, provided: {", ".join(s2_order)}, available: {", ".join(self.s2_categories)}')
    
        self.densities = self._get_grid_and_densities(self.data_in, self.ylabel, self.s1, self.s2)
    
    
    def __repr__(self) -> str:
        return (
            f"SplitViolin(Numeric variable: {self.ylabel}\n "
            f"Split 1: {self.s1}, dtype: {self.s1_dtype}, levels: {len(self.s1_categories)}\n"
            f"Split 2: {self.s2}, dtype: {self.s2_dtype}, levels: {len(self.s2_categories)}\n "
            f"Total data points: {len(self.data_in)})"
        )

    def _get_grid_and_densities(self, data: pd.DataFrame, x:str, s1:str, s2:str) -> pd.DataFrame:

        return data.groupby([s1, s2]).apply(self._get_curves, x=x)
                
      
    def _get_curves(self, grouped_df:pd.DataFrame, x:str):       
        xvals = grouped_df[x].values
        support = _kde_support(xvals)
        kde = _fit_kde(xvals)
        density = kde(support)
        return pd.DataFrame({'grid':support, 'density':density})



# These are mostly helper functions/classes

def _kde_support(x: np.ndarray, 
                fit_kw: dict = None,
                **support_kwargs):
    """Create a 1D grid of evaluation points."""
    kde = _fit_kde(x, fit_kw=fit_kw)
    bw = np.sqrt(kde.covariance.squeeze())
    grid = _define_support_grid(x, bw, **support_kwargs)
    
    return grid

def _fit_kde(x:np.ndarray, 
             fit_kw: dict = None):
    """Fit a gaussian KDE to numeric variable and adjust bandwidth."""
    fit_props = {"bw_method": "scott"}
    
    if fit_kw:
        fit_props.update(fit_kw)
        
    kde = gaussian_kde(x, **fit_props)
    kde.set_bandwidth(kde.factor * 1)

    return kde

def _define_support_grid(x:np.ndarray, 
                  bw:float, 
                  cut:int=3, 
                  clip:tuple = None, 
                  gridsize:int = 200):
    """Create the grid of evaluation points depending for vector x."""
    if clip is None:
        clip = None, None
    clip_lo = -np.inf if clip[0] is None else clip[0]
    clip_hi = +np.inf if clip[1] is None else clip[1]
    gridmin = max(x.min() - bw * cut, clip_lo)
    gridmax = min(x.max() + bw * cut, clip_hi)
    
    return np.linspace(gridmin, gridmax, gridsize)

