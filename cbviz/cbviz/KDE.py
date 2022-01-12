# import
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


"""Functions for univariate distributions"""

class SplitViolin:
    
    """Container for producing a split violin plot and compute associated statistics"""

    def __init__(self, data:pd.DataFrame) -> None:
        pass





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

