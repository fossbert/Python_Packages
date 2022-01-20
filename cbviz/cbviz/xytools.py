# computing
from collections import namedtuple
from multiprocessing.sharedctypes import Value
from os import name
from warnings import warn
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, ConnectionPatch
from adjustText import adjust_text

# stats
from scipy.stats import pearsonr

# utils
from .utils import DataNum
from itertools import combinations


"""Functions for illustrating relationships between two numeric variables"""


class XYpairs:

    def __init__(self, 
                data: pd.DataFrame, 
                lower_tri: bool = True) -> None:


        self.data = DataNum(data)
        self.lower_tri = lower_tri
        self.gridn = self.data.ncols - 1
        self.combos = list(combinations(self.data.var_names, 2))
        self.grid = self._set_up_grid(self.gridn, self.lower_tri)

    def __repr__(self) -> str:

        return f"XYpairs(Number of columns: {self.data.ncols}, Observations: {len(self.data.df)})"
    
    def _set_up_grid(self, gridn, lower_tri: bool):
        
        cols, rows = np.meshgrid(range(gridn), range(gridn))
        
        if lower_tri:
            colindex = [y for x in [col[:i+1] for i, col in enumerate(cols)] for y in x]
            rowindex = [y for x in [row[:i+1] for i, row in enumerate(rows)] for y in x]

            return sorted([(i, j) for i, j in zip(rowindex, colindex)], reverse=True)

        else: 
            colindex = [y for x in [col[i:] for i, col in enumerate(cols)] for y in x]
            rowindex = [y for x in [row[i:] for i, row in enumerate(rows)] for y in x]

            return sorted([(i, j) for i, j in zip(rowindex, colindex)])
        
    def get_gridspec(self, fig=None, **gridspec_kwargs):
        
        if fig is None:
            fig = plt.gcf()
    
        return fig.add_gridspec(nrows= self.gridn , ncols= self.gridn , **gridspec_kwargs)
    
    def get_pairs(self, **xyview_kwargs):
        
        for combo, pos in zip(self.combos, self.grid):
            
            yield XYviewGrid(list(combo), pos, XYview(self.data.df[list(combo)], **xyview_kwargs))    
            
    
class XYview:
    
    """A class to implement tools for examining the relationship of two numeric variables which are provided in a DataFrame. 
    There is an option of producing labels of certain data points. A variable number of keyword arguments will be gathered and 
    can be passed to a call to the pyplot.scatter function. The data points to be shown must be present in the DataFrame's Index.

    """

    def __init__(self, 
                    data:pd.DataFrame,  
                    highlight: list = None,
                    **scatter_kwargs):

        """[summary]

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
        """
        
        
        self.data = DataNum(data, ncols=2)
        self.xlabel, self.ylabel = self.data.var_names
        self.x = self.data.df[self.xlabel].values
        self.y = self.data.df[self.ylabel].values
        
        # stats
        self.pearson, _ = pearsonr(self.x, self.y)
        self.slope, self.intercept  = np.polyfit(self.x, self.y, 1)
        
        # scatter keywords
        self.scatter_kw = {'c': '0.5', 'alpha':0.5, 'linewidth':0, 's':200/len(self.data.df)}
        if len(scatter_kwargs)>0:
            for key, val in scatter_kwargs.items():
                self.scatter_kw.update({key:val})

        # Labels
        self.highlight = highlight
        self.pearson_label_props =   {'loc':0, 'handlelength':0, 'handletextpad':0, 
        "frameon":False, 'fontsize':'x-small', 'labelcolor':'0.15'}

        # Lines
        self.line_props =  {'lw':0.5, 'ls':':', 'color':'.15', 'zorder':-1}

        if self.highlight:
            self.data_highlight = self.data.df[self.data.df.index.isin(highlight)]
            if len(self.data_highlight)==0:
                warn('Highlight list not found in DataFrame index. Returning empty DataFrame.', RuntimeWarning)
            

    def __repr__(self) -> str:
        return (
            f"XYview(X: {self.xlabel}, Y: {self.ylabel}\n"
            f"Observations: {len(self.data.df)}, Pearson r: {self.pearson:.2f})"
            )

    
    
    def add_correlation(self, ax=None, **legend_kwargs):
        
        if ax is None:
            ax = plt.gca()
        
        if len(legend_kwargs)>0:
            for k,v in legend_kwargs.items():
                self.pearson_label_props.update({k:v})
                
        ax.legend([Patch(color='w')], [f'r: {self.pearson:.2f}'], **self.pearson_label_props)
        
        return ax
        
    def add_reg_line(self, ax=None, **line_kwargs):
        
        if ax is None:
            ax = plt.gca()
        
        if len(line_kwargs)>0:
            for k,v in line_kwargs.items():
                self.line_props.update({k:v})

        ax.plot(self.x, self.x*self.slope + self.intercept, **self.line_props)
        
        return ax

    def add_xy_line(self,  ax=None, **line_kwargs):

        if ax is None:
            ax = plt.gca()
        
        if len(line_kwargs)>0:
            for k,v in line_kwargs.items():
                self.line_props.update({k:v})
        
        ax.plot(self.x, self.x*self.slope + self.intercept, **self.line_props)
        
        return ax
        

    def label_dots(self, ax=None, adjust=False, **text_kwargs):
        """[summary]

        Raises:
            AttributeError: [If called on an instance of XYview without the highlight attribute]

        Returns:
            [type]: [description]
        """

        if ax is None:
            ax = plt.gca()

        try:
            
            txts = [ax.text(getattr(row, self.xlabel), getattr(row, self.ylabel), row.Index, **text_kwargs) for row in self.data_highlight.itertuples()]
            if adjust:
                adjust_text(txts)
            return ax
       
        except AttributeError:
            print('No labelling without the highlight attribute!')
            raise 
       
       
    def label_xy(self, ax=None, outer=False, **text_kwargs):
        
        if ax is None:
            ax = plt.gca()
                
        ax.set_xlabel(self.xlabel, **text_kwargs)
        ax.set_ylabel(self.ylabel, **text_kwargs)
        
        if outer:
            ax.xaxis.tick_top()
            ax.yaxis.tick_right()
            
        return ax


class XYzoom:
    
    def __init__(self, data, xrange:tuple, yrange:tuple) -> None:
        
        self.data = DataNum(data, ncols=2)
        self.xlabel, self.ylabel = self.data.var_names
        self.xmin, self.ymin = self.data.df.min().values
        self.xmax, self.ymax = self.data.df.max().values
    
        self.zoom_xmin, self.zoom_xmax = self._deduce_range(xrange, 'x')
        self.zoom_ymin, self.zoom_ymax = self._deduce_range(yrange, 'y')
        
        
    def __repr__(self) -> str:
        return  (f"XYzoom(X: {self.xlabel}, Y: {self.ylabel}\n"
                f"Observations: {len(self.data.df)}\n"
                f"Zoom area: X ({self.zoom_xmin:.2f} -> {self.zoom_xmax:.2f}), Y ({self.zoom_ymin:.2f} -> {self.zoom_ymax:.2f})")
        
    def _deduce_range(self, range: tuple, which:str):
           
        combo = ''.join([str(i) for i in range])
         
        if combo.startswith('min') & combo.endswith('max'):
            
            return (self.xmin, self.xmax) if which == 'x' else (self.ymin, self.ymax)
        
        elif combo.startswith('min'):
            
            upper = range[1]
            
            return (self.xmin, upper) if which == 'x' else (self.ymin, upper)
        
        elif combo.endswith('max'):
            
            lower = range[0]
        
            return (lower, self.xmax) if which == 'x' else (lower, self.ymax)
    
        else:
            
            try:
                
                return [float(i) for i in range]
     
            except ValueError:
                
                valid = ['(min, max)', '(min, 2)', '(0, max)', '(-3, 2)']
                print(f"Valid examples: {', '.join(valid)} - got: {combo}") 
                raise
          
             

             
        
#     def _filter(self, data, xrange, yrange):
        
#         query = f"{self."

#         xy_rect = sub[['sel', 'tra']].agg(['min', 'max']).loc['min'].values
# rect_w, rect_h = sub[['sel', 'tra']].agg(['min', 'max']).apply(lambda x: abs(x[0]-x[1]), axis=0).values

## Data containers

XYviewGrid = namedtuple('XYviewGrid', 'combo position XYview')