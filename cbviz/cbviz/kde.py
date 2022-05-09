# pillars of work
import numpy as np
import pandas as pd

# tools and stats
from collections import namedtuple
from functools import singledispatchmethod

# Plotting
from matplotlib.lines import Line2D

# stats
from scipy.stats import gaussian_kde

# documentation 
from typing import Union

from .utils import DataMix

"""Functions for illustrating gaussian kernel density estimates"""


class KDE:
    
    """ Basic class implementing tools for visualizing and annotating kernel density estimates. 

    """
    
    def __init__(self, 
                 data:Union[pd.Series, np.ndarray],
                 fit_kwargs:dict=None,
                 grid_kwargs:dict=None) -> None:
        
        if isinstance(data, np.ndarray):
            print('Provided np.ndarray -> defaulting to RangeIndex')
            data = pd.Series(data)
            
        self.data = data.dropna()
        self.kde = _fit_kde(self.data.values, fit_kwargs) # important to find index or values
        self.curve = _get_curve(self.data, None, fit_kwargs, grid_kwargs)
        
        
    def __repr__(self) -> str:
        return (
            f"KDE(Total data points: {len(self.data)},\n"
            f"gridsize: {len(self.curve)}"
        )

    @singledispatchmethod
    def find(self, position):
        """Allows to find coordinates for either indices (provided as strings) or numeric values from the 
        data. It will return a tuple of a x and y value for further processing. 
        """
        print('Please specify position as either a float, integer or string.')

    @find.register(float)
    @find.register(int)
    def find_num(self, position):
        return (position, self.kde(position)[0])

    @find.register
    def _(self, position: str):
        pos_numeric = self.data[position]
        return (pos_numeric, self.kde(pos_numeric)[0])
    
    @property
    def x(self):
        return self.curve.grid #convenience
    
    @property
    def y(self):
        return self.curve.density #convenience


class KDE2D:

    """ Class implementing tools for visualizing and annotating 2-dimensional kernel density estimates. """
   
    def __init__(self, 
                 x:Union[pd.Series, np.ndarray],
                 y:Union[pd.Series, np.ndarray],
                 fit_kwargs:dict=None,
                 grid_kwargs:dict=None) -> None:
        
        if isinstance(x, np.ndarray):
            x = pd.Series(x)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            
        self.x = x.dropna()
        self.y = y.dropna()
        self.kde = _fit_kde(self.x, self.y, fit_kwargs=fit_kwargs) 
        self.bw1, self.bw2 = np.diag(np.sqrt(self.kde.covariance.squeeze()))
        self.grid_props = {'cut':3, 'gridsize':200, 'clip':None}
        if grid_kwargs:
            for k, v in grid_kwargs.items():
                if k in self.grid_props:
                    self.grid_props.update({k:v})
                else:
                    print(f'Ignoring following key: {k}')

        self.support = [_define_support_grid(i, bw, **self.grid_props) for i, bw in zip([self.x, self.y], [self.bw1, self.bw2])]
        self.X, self.Y = np.meshgrid(*self.support)
        self.Z = self.kde([self.X.ravel(), self.Y.ravel()]).reshape(self.X.shape)

    @property
    def data(self):
        return self.X, self.Y, self.Z

#     kde = gaussian_kde([m1, m2])
# bw1, bw2 = np.diag(np.sqrt(kde.covariance.squeeze()))
# support = (_define_support_grid(i, bw) for i, bw in zip([m1, m2], [bw1, bw2]))
# X, Y = np.meshgrid(*support)
# Z = kde([X.ravel(), Y.ravel()]).reshape(X.shape)
# fig, ax = plt.subplots()

# ax.contour(X, Y, Z, colors='C0', linewidths=0.5, zorder=-1)
# ax.scatter(m1, m2, alpha=0.6)

class Ridge:
    
    """Convenience class to produce a series of density curves for plotting density curves 
    broken down by one categorical variable (=s1). Usually set up along the y-axis. 
    
    Example use case: 
    rp = Ridge(data, s1_order=['WT','Balanced', 'Minor', 'Major'], scale_factor=1.2)
    
    for kde in rp.get_kdes():

        ax.fill_betweenx(kde.density, kde.grid, facecolor=kde.color, alpha=0.3)
        ax.plot(kde.mode.ycoords, kde.mode.xcoords, lw=0.1, c='k')

    ax.set_yticks(*rp.get_s1_ticks())
    ax.set_xlabel(rp.ylabel)
    [ax.axhline(i, ls=':', lw=0.25, c='0.15') for i in rp.get_s1_ticks()[0]]
    
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


def _get_curve(data:Union[pd.DataFrame, pd.Series], 
               x:str = None, 
               fit_kwargs:dict=None,
               grid_kwargs:dict=None):       
    
    """[This function is the main work horse to go from a series of numeric values to 
    a density curve with support grid which it will provide in a DataFrame. 
    It expects either a DataFrame - grouped or not - or in the simplest case a Series object.]

    Returns:
        [pd.DataFrame]: [DataFrame with support grid and density values across this grid.]
    """
    
    xvals = data[x].values if x else data.values
    
    # fit kde and squeeze bandwidth
    kde = _fit_kde(xvals, fit_kwargs)
    bw = np.sqrt(kde.covariance.squeeze())
    
    # get support grid, pass further arguments if provided
    grid_props = {'cut':3, 'gridsize':200, 'clip':None}
    if grid_kwargs:
        for k, v in grid_kwargs.items():
            if k in grid_props:
                 grid_props.update({k:v})
            else:
                print(f'Ignoring following key: {k}')
                
    grid = _define_support_grid(xvals, bw, **grid_props)
    # density over grid
    density = kde(grid)
    
    return pd.DataFrame({'grid':grid, 'density':density})

def _fit_kde(*arrays, fit_kwargs: dict=None):
    """Fit a gaussian KDE to numeric variable and adjust bandwidth."""
   
    fit_props = {"bw_method": "scott"}
    
    if fit_kwargs:
        fit_props.update(fit_kwargs)

    if (n_arr:=len(arrays))>2:
        raise ValueError(f'Can only accomodate two input arrays, got {n_arr}')
        
    kde = gaussian_kde(dataset=arrays, **fit_props)
    kde.set_bandwidth(kde.factor * 1)

    return kde


def _define_support_grid(x:np.ndarray, bw:float, cut:int=3, gridsize:int = 200, clip:tuple = None):
    
    """Create the grid of evaluation points depending for vector x."""


    clip_lo = clip[0] if clip else -np.inf 
    clip_hi = clip[1] if clip else +np.inf 

    gridmin = max(x.min() - bw * cut, clip_lo)
    gridmax = min(x.max() + bw * cut, clip_hi)
    
    return np.linspace(gridmin, gridmax, gridsize)
 

