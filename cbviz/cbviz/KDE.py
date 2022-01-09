# import
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from collections import namedtuple

"""Functions for univariate distributions"""
    
def _fit(fit_data:np.ndarray, 
         fit_kw: dict = None):
    fit_props = {"bw_method": "scott"}
    
    if fit_kw:
        fit_props.update(fit_kw)
        
    kde = gaussian_kde(fit_data, **fit_props)
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

def _define_support_univariate(x: np.ndarray, 
                               fit_kw: dict = None):
    """Create a 1D grid of evaluation points."""
    kde = _fit(x, fit_kw=fit_kw)
    bw = np.sqrt(kde.covariance.squeeze())
    grid = _define_support_grid(x, bw)
    
    return grid