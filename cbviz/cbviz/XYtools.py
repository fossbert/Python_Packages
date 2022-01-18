# computing
from distutils.errors import LibError
from nis import cat
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# stats
from scipy.stats import pearsonr


class XYview:
    
    def __init__(self, data:pd.DataFrame,  
                    highlight: list = None,
                    pearson_label_kw: dict = None,
                    **scatter_kwargs):
        
        if isinstance(data, pd.DataFrame):
            
            self.dtypes_in = np.array([data[col].dtype.name.rstrip('_12346') for col in data.columns]) 

            if all(self.dtypes_in == np.array(['float', 'float'])):
                self.data_in = data.dropna().copy()
                self.xlabel, self.ylabel = data.columns.to_list()
                self.x = self.data_in[self.xlabel].values
                self.y = self.data_in[self.ylabel].values

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
                raise ValueError(f'Only floats allowed, found {", ".join(self.dtypes_in)}')

        else:
            raise ValueError('Need a pandas DataFrame')



        if highlight:
            self.data_highlight = self.data_in[self.data_in.index.isin(highlight)]
            

    def __repr__(self) -> str:
        return (
            f"XYview(X: {self.xlabel}, Y: {self.ylabel}\n"
            f"Observations: {len(self.data_in)}, Pearson r: {self.pearson:.2f})"
            )
    
    def get_pearson_label(self, text_kw: dict=None):

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

    def label_dots(self, text_kw: dict = None):

        try:
            return ((getattr(row, self.xlabel), getattr(row, self.ylabel), row.Index) for row in self.data_highlight.itertuples())
        except:
            raise AttributeError('No labelling without the highlight attribute!') 
       

