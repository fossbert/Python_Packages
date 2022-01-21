# computing
from collections import namedtuple
from multiprocessing.sharedctypes import Value
from os import name
from warnings import warn
import matplotlib
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


    """Container to illustate the XY-relationship for all combinations of numeric columns in a DataFrame.
     A ncol-1 grid is set up and the user can choose whether to fill the upper or lower triangle, respectively. 
     Most of heavy lifting is carried out by the XYview class which - together with information on the two variables
     and their position in the grid is provided by a call to the ged_pairs() method. 
    """

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
            
            yield XYpair(list(combo), pos, XYview(self.data.df[list(combo)], **xyview_kwargs))    
            
    
class XYview:
    
    """A class to implement tools for examining the relationship of two numeric variables 
    which are provided in a DataFrame. There is an option of producing labels of certain data points.
    A variable number of keyword arguments will be gathered and can be passed to a call to the pyplot.scatter function.
    The data points to be shown must be present in the DataFrame's Index.

    Example: 
    metabolites = ['L-Cystine', 'Cysteine', 'NADH']
    all = pd.read_csv('./Metabolite-logFC-Signatures.csv', index_col=0)
    df = all.iloc[:, [0,2]].copy()
    
    cond = df.index.isin(metabolites)
    xv = XYview(df, highlight=metabolites, s=np.where(cond, 40, 10), c=np.where(cond, 'r', '.5'), alpha=1)
    plt.style.use('cviz')

    fig, ax = plt.subplots(figsize=(3,3))

    ax.scatter(xv.x, xv.y, **xv.scatter_kw)
    xv.add_correlation(fontsize='small')
    xv.add_reg_line(color='cornflowerblue', lw=1)
    xv.label_dots(adjust=True, fontsize='small')
    xv.label_xy()
    """

    def __init__(self, 
                    data:pd.DataFrame,  
                    highlight: list = None,
                    **scatter_kwargs):    
        
        self.data = DataNum(data, ncols=2)
        self.xlabel, self.ylabel = self.data.var_names
        self.x = self.data.df[self.xlabel].values
        self.y = self.data.df[self.ylabel].values
        
        # stats
        self.pearson, _ = pearsonr(self.x, self.y)
        self.slope, self.intercept  = np.polyfit(self.x, self.y, 1)
        
        # scatter keywords
        self.scatter_kw = {'c': '0.5', 'alpha':0.5, 'linewidth':0, 's':2000/len(self.data.df)}
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
        
        
        ax.plot(ax.get_xlim(), ax.get_ylim(), **self.line_props)
        
        return ax
        

    def label_dots(self, ax=None, adjust=False, adjust_kwargs:dict=None, **text_kwargs):
        """[summary]

        Raises:
            AttributeError: [If called on an instance of XYview without the highlight attribute]

        Returns:
            [matplotlib.pyplot.axes]: [axes with texts added, possibly adjusted for overlap]
        """

        if ax is None:
            ax = plt.gca()

        try:
            
            txts = [ax.text(getattr(row, self.xlabel), getattr(row, self.ylabel), row.Index, **text_kwargs) for row in self.data_highlight.itertuples()]
            if adjust:
                adjust_props = {'ax':ax}
                if adjust_kwargs:
                    adjust_props.update(adjust_kwargs)
                adjust_text(txts, **adjust_props)
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


class XYzoom(XYview):
    
    """A class to implement a zoom option on a particular area of a XY relationship. Inherits from XYview. 
    
   
    Example: 
    
    xz = XYzoom(df, ('min', 0), ('min', 0))
    top5 = xz.zoom_data.mean(1).nsmallest(5).index.to_list()
    cond = xz.data.df.index.isin(top5)

    # Here, ax1 will be the zoomed in axes. So we connect to the left. 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4,2))

    ax2.scatter(xz.x, xz.y, **xz.scatter_kw)
    xz.add_xy_line(ax2)
    xz.add_rect(ax=ax2)
    ax1.scatter(xz.x, xz.y, s=np.where(cond, 20, 5), c=np.where(cond, 'r', '.5'), alpha=0.5, linewidths=0)
    xz.connect(ax2, ax1, 'left', lw=0.5, ls=':')
    xz.label_xy(ax=ax1, outer=False, fontsize='x-small')
    xz.label_dots(top5, ax=ax1, adjust=True, adjust_kwargs={'arrowprops':{'arrowstyle':'-', 'lw':0.2}}, fontsize=4)
    """

    def __init__(self, data, 
                 xrange:tuple, 
                 yrange:tuple,
                 zoom_margin:float=0.01, 
                 **scatter_kwargs,
                 ) -> None:
        
        super().__init__(data, **scatter_kwargs)
        self.xmin, self.ymin = self.data.df.min().values
        self.xmax, self.ymax = self.data.df.max().values
        self.zoom_margin = zoom_margin * np.ptp(self.data.df.values)
        self.xrange = self._deduce_range(xrange, 'x')
        self.xrange_zoom = self._adjust_range(self.xrange)
        self.yrange = self._deduce_range(yrange, 'y')
        self.yrange_zoom = self._adjust_range(self.yrange)
        self.rect_width, self.rect_height = self.xrange.max-self.xrange.min, self.yrange.max-self.yrange.min
        self.zoom_data = self._filter()
        
    def __repr__(self) -> str:
        return  (f"XYzoom(X: {self.xlabel}, Y: {self.ylabel}\n"
                  f"Zoom area: X ({self.xrange.min:.2f} -> {self.xrange.max:.2f}), Y ({self.yrange.min:.2f} -> {self.yrange.max:.2f})\n"
                f"Observations total: {len(self.data.df)}, Observation(s) in zoom: {len(self.zoom_data)})"
               )
        
    def _deduce_range(self, range: tuple, which:str):  
        
        combo = ''.join([str(i) for i in range])         
        if combo.startswith('min') & combo.endswith('max'):            
            return AxRange(self.xmin, self.xmax) if which == 'x' else AxRange(self.ymin, self.ymax)
        
        elif combo.startswith('min'):            
            upper = range[1]            
            return AxRange(self.xmin, upper) if which == 'x' else AxRange(self.ymin, upper)
        
        elif combo.endswith('max'):            
            lower = range[0]        
            return AxRange(lower, self.xmax) if which == 'x' else AxRange(lower, self.ymax)
    
        else:            
            try:
                return AxRange(*[float(i) for i in range])
            except ValueError:
                valid = ['(min, max)', '(min, 2)', '(0, max)', '(-3, 2)']
                print(f"Valid examples: {', '.join(valid)} - got: {combo}") 
                raise
          
          
    def _adjust_range(self, range):
        
        adjust = [val-self.zoom_margin if i==0 else val+self.zoom_margin for i, val in enumerate(range)]
    
        return AxRange(*adjust)
    
    def _filter(self):
        
        query = (f"({self.xlabel}>={self.xrange.min})&({self.xlabel}<={self.xrange.max})"
                 f"&({self.ylabel}>={self.yrange.min})&({self.ylabel}<={self.yrange.max})")

        zoom_data = self.data.df.query(query).copy()
        if not len(zoom_data):
            raise ValueError('There is no data to see in this zoom area, please inspect first.')
        else:
            return zoom_data
        
    def add_rect(self, ax=None, **rect_kwargs) -> matplotlib.pyplot.axes:
        
        """[Adds a rectangle illustrating the zoom area to the main, non-zoomed axes. Axes must 
        should be specified, otherwise current axes is used.]

        Returns:
            [matplotlib.pyplot.axes]: [Modifed axes]
        """
    
        rect_props = {'alpha':0.1, 'color':'C0', 'lw':0}
        
        if len(rect_kwargs):
            for k, v in rect_kwargs.items():
                rect_props.update({k:v})
    
        if ax is None:
            ax = plt.gca()
    
        rect = Rectangle((self.xrange.min, self.yrange.min), self.rect_width, self.rect_height,  **rect_props)
        
        ax.add_artist(rect)

        return ax
    
    def connect(self, main_ax, zoom_ax, loc:str, fig=None, **line_kwargs):
        
        """[Adds two connection lines to figure, leading the viewer's eye from the rectangle in the non-zoomed axes 
        to the respective corners of the zoomed in axes. Furthermore, it will add the zoom effect by 
        resetting the x- and y-limits for the zoomed axes. User must provide the location of the zoomed in axes 
        with respect to the main axes.]

        Returns:
            Nothing, fig and zoom_axes are modified
        """
        
        if fig is None:
            fig = plt.gcf()
        
        # setup the corners
        lower_left, upper_left = [(x, y) for x in [self.xrange.min] for y in self.yrange]
        lower_right, upper_right = [(x, y) for x in [self.xrange.max] for y in self.yrange]
        lower_left_zoom, upper_left_zoom = [(x, y) for x in [self.xrange_zoom.min] for y in self.yrange_zoom]
        lower_right_zoom, upper_right_zoom = [(x, y) for x in [self.xrange_zoom.max] for y in self.yrange_zoom]
      
        
        conn_options = {'left': ConnInfo(ConnLine(lower_left, lower_right_zoom),ConnLine(upper_left, upper_right_zoom)),
                    'top': ConnInfo(ConnLine(upper_right, lower_right_zoom), ConnLine(upper_left, lower_left_zoom)), 
                    'right':ConnInfo(ConnLine(lower_right, lower_left_zoom), ConnLine(upper_right, upper_left_zoom)),  
                    'bottom': ConnInfo(ConnLine(lower_right, upper_right_zoom), ConnLine(lower_left, upper_left_zoom))}
        
        # Zoom
        zoom_ax.set_xlim(self.xrange_zoom)
        zoom_ax.set_ylim(self.yrange_zoom)
        
        # Connect
        if conn_lines:=conn_options.get(loc):
                
            con1 = ConnectionPatch(xyA=conn_lines.line1.acoords, coordsA=main_ax.transData, 
                            xyB=conn_lines.line1.bcoords, coordsB=zoom_ax.transData, **line_kwargs)

            fig.add_artist(con1)
            
            con2 = ConnectionPatch(xyA=conn_lines.line2.acoords, coordsA=main_ax.transData, 
                            xyB=conn_lines.line2.bcoords, coordsB=zoom_ax.transData, **line_kwargs)
            
            fig.add_artist(con2)
        
        else:
            raise ValueError(f"{loc} is not a valid option, choose from {', '.join(list(conn_options.keys()))}")
    
    def label_dots(self, highlight:list, ax=None, adjust=False, adjust_kwargs:dict=None, **text_kwargs):
        
        """[Labels dots for a provided list of labels. Note this is different from XYview's label_dots method,
        ]

        Returns:
            [type]: [description]
        """
        
        lbl_data = self.zoom_data[self.zoom_data.index.isin(highlight)].copy()
        
        if len(lbl_data)==0:
            warn('Highlight list not found in zoom DataFrame index. Returning empty DataFrame.', RuntimeWarning)
            
        if ax is None:
            ax = plt.gca()
    
        txts = [ax.text(getattr(row, self.xlabel), getattr(row, self.ylabel), row.Index, **text_kwargs) for row in lbl_data.itertuples()]
        if adjust:
            adjust_props = {'ax':ax}
            if adjust_kwargs:
                adjust_props.update(adjust_kwargs)
            adjust_text(txts, **adjust_props)
        return ax
    

## Data containers and helpers, respectively

XYpair = namedtuple('XYpair', 'combo position XYview')
AxRange = namedtuple('AxRange', 'min max')
ConnInfo = namedtuple('ConnInfo', 'line1 line2')
ConnLine = namedtuple('ConnLine', 'acoords bcoords')

        