# pillars of work
import numpy as np
import pandas as pd

# tools and stats
from itertools import product
from collections import namedtuple


# Plotting
from matplotlib.lines import Line2D

# stats
from scipy.stats import gaussian_kde

# documentation 
from typing import Union

from .utils import DataMix

"""Functions for illustrating gaussian kernel density estimates"""


class Ridge:
    
    
    """[summary]

    Raises:
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        AssertionError: [description]

    Returns:
        [type]: [description]

    Yields:
        [type]: [description]
    """

    def __init__(self, 
                 data:pd.DataFrame,  
                scale_factor:int = 1,   
                minsize:int = 5,    
                s1_order: list = None) -> None:
        
        
        self.data = DataMix(data, ncat=1, minsize=minsize)
        self.ylabel, self.s1 = self.data.var_names
        _, self.s1_dtype = self.data.dtypes
        self.scale_factor = scale_factor
        
        # Second checks the split variable: 
        # 1.) Do they need conversion ? 
        # 2.) s1 needs at least one level
        # 3.) If order is provided, reorder if provided levels match
        if self.s1_dtype == 'string':
            self.data.df[self.s1] = self.data.df[self.s1].astype('category')
            self.s1_dtype = 'categorical'

        self.s1_categories = self.data.df[self.s1].cat.categories.to_list()
        
        if len(self.s1_categories) < 1:
            raise ValueError(f'There is not a single level for {self.s1}') 
        
        if s1_order:
             if all([s1_lvl in self.s1_categories for s1_lvl in s1_order]):
                 self.data.df[self.s1] = self.data.df[self.s1].cat.reorder_categories(s1_order)
                 self.s1_categories = s1_order
             else:
                 raise ValueError(f'Could not align levels, provided: {", ".join(s1_order)}, available: {", ".join(self.s1_categories)}')
    
        # Once we made it past all these checks, we can start with some calculations
        # sharing is easy between Ridge and SplitViolin
        self.densities = _get_grids_and_densities(self.data.df, self.ylabel, self.s1)

    def __repr__(self) -> str:
        return (
            f"Ridge(Numeric variable: {self.ylabel}\n "
            f"Split 1: {self.s1}, levels: {len(self.s1_categories)}\n"
            f"Total data points: {len(self.data.df)})"
        )

    def get_kdes(self, colors: tuple = None):
        """Generates KDE information for each subgroup and provides them as namedtuple."""
    
        for i, (s1, subdf1) in enumerate(self.densities.groupby(level=0)):

            grid = subdf1['grid'].values
            density = subdf1['density'].div(subdf1['density'].max()).values
            density = density + i * self.scale_factor
            mode_y = np.repeat(grid[density.argmax()], 2)
            mode_x = np.array([np.min(density), np.max(density)])
            
            color = colors[i] if colors else '0.5'
            
            yield KdeData(s1, color, grid, density, Mode(mode_x, mode_y))
            
    def get_s1_ticks(self):
    
        return [np.arange(0, len(self.s1_categories))*self.scale_factor, self.s1_categories]



class SplitViolin:
    
    """Container for producing a split violin plot and compute associated statistics."""

    def __init__(self, data:pd.DataFrame,  
                scale_factor:int = 2,   
                minsize:int = 5,    
                s1_order: list = None, 
                s2_order: list = None) -> None:
        
        self.data = DataMix(data, ncat=2, minsize=minsize)
        self.ylabel, self.s1, self.s2 = self.data.var_names
        _, self.s1_dtype, self.s2_dtype = self.data.dtypes
        self.scale_factor = scale_factor    
    
        # Second checks the split variables: 
        # 1.) Do they need conversion ? 
        # 2.) s2 can only have two levels, s1 needs at least one level
        # 3.) If order is provided, reorder if provided levels match
        if self.s1_dtype == 'string':
            self.data.df[self.s1] = self.data.df[self.s1].astype('category')
            self.s1_dtype = 'categorical'
            
        if self.s2_dtype == 'string':
            self.data.df[self.s2] = self.data.df[self.s2].astype('category')
            self.s2_dtype = 'categorical'
        
        self.s1_categories = self.data.df[self.s1].cat.categories.to_list()
        self.s2_categories = self.data.df[self.s2].cat.categories.to_list()
        
        if s2_ncat := len(self.s2_categories) != 2:
            raise ValueError(f'Need 2 levels for {self.s2}, found {s2_ncat}') 

        if len(self.s1_categories) < 1:
            raise ValueError(f'There is not a single level for {self.s1}') 
        
        if s1_order:
             if all([s1_lvl in self.s1_categories for s1_lvl in s1_order]):
                 self.data.df[self.s1] = self.data.df[self.s1].cat.reorder_categories(s1_order)
                 self.s1_categories = s1_order
             else:
                 raise ValueError(f'Could not align levels, provided: {", ".join(s1_order)}, available: {", ".join(self.s1_categories)}')
        
        if s2_order:
            if all([s2_lvl in self.s2_categories for s2_lvl in s2_order]):
                self.data.df[self.s2] = self.data.df[self.s2].cat.reorder_categories(s2_order)
                self.s2_categories = s2_order
            else:
                raise ValueError(f'Could not align levels, provided: {", ".join(s2_order)}, available: {", ".join(self.s2_categories)}')
    
        # Once we made it past all these checks, we can start with some calculations
        self.densities = _get_grids_and_densities(self.data.df, self.ylabel, self.s1, self.s2)

    def __repr__(self) -> str:
        return (
            f"SplitViolin(Numeric variable: {self.ylabel}\n "
            f"Split 1: {self.s1}, levels: {len(self.s1_categories)}\n"
            f"Split 2: {self.s2}, levels: {len(self.s2_categories)}\n "
            f"Total data points: {len(self.data.df)})"
        )

    def get_violins(self, colors: tuple = ('C0', 'C1')):
        """Generates KDE information for each subgroup and provides them as namedtuple."""

        for i, (s1, subdf1) in enumerate(self.densities.groupby(level=0)):
            for j, (s2, subdf2) in enumerate(subdf1.groupby(level=1)):

                grid = subdf2['grid'].values
                name = '_'.join([s1, s2])
                density = subdf2['density'].div(subdf2['density'].max()).values
                if j%2==0:
                    density =  density*-1 + i * self.scale_factor
                    color = colors[0]
                    mode_y = np.repeat(grid[density.argmin()], 2)
                else:
                    density =  density + i * self.scale_factor
                    color = colors[1]
                    mode_y = np.repeat(grid[density.argmax()], 2)
                mode_x = np.array([np.min(density), np.max(density)])
              
                yield KdeData(name, color, grid, density, Mode(mode_x, mode_y))

    def get_s1_ticks(self):
        
        return [range(0, len(self.s1_categories)*self.scale_factor, 2), self.s1_categories]
    
    
    def get_s2_legend(self, colors: tuple = ('C0', 'C1'), **line2d_kwargs):
        
        line2d_props = {'marker':'s', 'color':'w', 'markersize':6}
        
        if len(line2d_kwargs)>0:
            for k,v in line2d_kwargs.items():
                line2d_props.update({k:v})
                
        return [Line2D([0], [0], markerfacecolor=col, label=cat, **line2d_props) for cat, col in zip(self.s2_categories, colors)]
        
        
    
####################################################
##                                                ##
##  These are helper functions/containers         ##
##                                                ##
####################################################

                                                
KdeData = namedtuple('KdeData', "name color grid density mode")

Mode = namedtuple('Mode', "xcoords ycoords")


def _get_grids_and_densities(data: pd.DataFrame, x:str, *split_variables) -> pd.DataFrame:

    """[This function takes a grouped DataFrame and applies the get_curve function to axis=0. 
    It will a indexed - possibly MultiIndexed - DataFrame containing grids and densities for each 
    of the split levels provided.]

    Returns:
        [pd.DataFrame]: [DataFrame indexed by split variables with support grids and densities.]
    
    """
    
    # Make sure split_variables are in data columns
    if all([split_var in data.columns for split_var in split_variables]):
        
        return data.groupby(list(split_variables)).apply(_get_curve, x=x)
    
    else:
        raise AssertionError(f"Split variables provided ({', '.join(split_variables)}) not found in data: {', '.join(data.columns)}")


def _get_curve(data:Union[pd.DataFrame, pd.Series], x:str = None):       
    
    """[This function is the main work horse to go from a series of numeric values to 
    a density curve with support grid which it will provide in a DataFrame. 
    It expects either a DataFrame - grouped or not - or in the simplest case a Series object.]

    Returns:
        [pd.DataFrame]: [DataFrame with support grid and density values across this grid.]
    """
    
    xvals = data[x].values if x else data.values
    support = _kde_support(xvals)
    kde = _fit_kde(xvals)
    density = kde(support)
    return pd.DataFrame({'grid':support, 'density':density})


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