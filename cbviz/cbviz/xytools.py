# computing
from warnings import warn
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# stats
from scipy.stats import pearsonr

"""Functions for illustrating relationships between two numeric variables"""


class PairGrid:

    def __init__(self, 
                data: pd.DataFrame, 
                lower_tri: bool = True,
                highlight: list = None,
                pearson_label_kw: dict = None,
                gridspec_kw: dict = None, 
                **scatter_kwargs) -> None:

        if isinstance(data, pd.DataFrame):

            err_out = "Need a DataFrame with three or more numeric (float) columns!"
            
            self.dtypes_in = np.array([data[col].dtype.name.rstrip('_12346') for col in data.columns])

            if len(self.dtypes_in) < 3:
                raise ValueError(err_out)
            
            self.ncols = len(self.dtypes_in)

            if all(self.dtypes_in == np.repeat('float', self.ncols)):
                self.data_in = data.copy()
                self.var_names = data.columns.to_list()
            else:
                raise ValueError(err_out)


    def __repr__(self) -> str:

        return f"PairGrid(Number of columns: {self.ncols}, Observations: {len(self.data_in)})"
    
class XYview:
    
    """A class to implement tools for examining the relationship of two numeric variables which are provided in a DataFrame. 
    There is an option of producing labels of certain data points. A variable number of keyword arguments will be gathered and 
    can be passed to a call to the pyplot.scatter function. The data points to be shown must be present in the DataFrame's Index.

    """

    def __init__(self, 
                    data:pd.DataFrame,  
                    highlight: list = None,
                    pearson_label_kw: dict = None,
                    **scatter_kwargs):

        """[summary]

        Raises:
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
        """
        
        if isinstance(data, pd.DataFrame):

            err_out = "Need a DataFrame with two numeric (float) columns!"
            
            self.dtypes_in = np.array([data[col].dtype.name.rstrip('_12346') for col in data.columns])

            if len(self.dtypes_in)!=2:
                raise ValueError(err_out)

            if all(self.dtypes_in == np.array(['float', 'float'])):
                self.data_in = data.dropna().copy()
                self.xlabel, self.ylabel = data.columns
                self.x = self.data_in[self.xlabel].values
                self.y = self.data_in[self.ylabel].values
            else:
                raise ValueError(err_out)

            # stats
            self.pearson, _ = pearsonr(self.x, self.y)
            self.slope, self.intercept  = np.polyfit(self.x, self.y, 1)
          
            # scatter keywords
            self.scatter_kw = {'c': '0.5', 'alpha':0.5, 'linewidth':0, 's':200/len(self.data_in)}
            if len(scatter_kwargs)>0:
                for key, val in scatter_kwargs.items():
                    self.scatter_kw.update({key:val})

            # Labels
            self.highlight = highlight
            self.pearson_label_props =   {'loc':0, 'handlelength':0, 'handletextpad':0, 
            "frameon":False, 'fontsize':'x-small', 'labelcolor':'0.15'}
            if pearson_label_kw:
                self.pearson_label_props.update(pearson_label_kw)

            # Lines
            self.line_props =  {'lw':0.5, 'ls':':', 'color':'.15', 'zorder':-1}

        else:
            raise ValueError('Need a pandas DataFrame')

        if self.highlight:
            self.data_highlight = self.data_in[self.data_in.index.isin(highlight)]
            if len(self.data_highlight)==0:
                warn('Highlight list not found in DataFrame index. Returning empty DataFrame.', RuntimeWarning)
            

    def __repr__(self) -> str:
        return (
            f"XYview(X: {self.xlabel}, Y: {self.ylabel}\n"
            f"Observations: {len(self.data_in)}, Pearson r: {self.pearson:.2f})"
            )
    
    def get_pearson_label(self):

        """[Function whose call should be passed to pyplot.legend via iterable unpacking. Will yield a legend
        to be placed using the 'best' location and puts the Pearson correlation coefficient on the axes. Should be 
        used together with self.pearson_label_props.]

        Returns:
            [tuple]: [containing the handles and labels for a text only legend.]
        """
        ptch = Patch(color='w')
        lbl = f'r: {self.pearson:.2f}'

        return ([ptch], [lbl])

    def get_reg_line(self, line_kw: dict = None):

        if line_kw:
            self.line_props.update(line_kw)

        return Line2D(self.x, self.x*self.slope + self.intercept, **self.line_props)

    def get_xy_line(self, line_kw: dict = None):

        if line_kw:
            self.line_props.update(line_kw)
        
        return Line2D(self.x, self.x*1, **self.line_props)

    def label_dots(self):
        """[summary]

        Raises:
            AttributeError: [If called on an instance of XYview without the highlight attribute]

        Returns:
            [type]: [description]
        """

        try:
            return ((getattr(row, self.xlabel), getattr(row, self.ylabel), row.Index) for row in self.data_highlight.itertuples())
        except:
            raise AttributeError('No labelling without the highlight attribute!') 
       

