# computing
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from adjustText import adjust_text

# stats
from scipy.stats import pearsonr


class XYview:
    
    def __init__(self, data:pd.DataFrame, 
                    highlight: list = None,
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
                self.pearson_label = self._get_pearson_label(self.pearson)
                self.pearson_label_props =   {'loc':0, 'handlelength':0, 'handletextpad':0, "frameon":False, 'fontsize':'x-small', 'labelcolor':'0.15'}
                self.slope, self.intercept  = np.polyfit(self.x, self.y, 1)
                self.reg_line = self._get_reg_line(self.slope, self.intercept)

                # scatter keywords
                self.scatter_kw = {'c': '0.5', 'alpha':0.5, 'linewidth':0, 's':100/len(self.data_in)}
                for key, val in scatter_kwargs.items():
                    self.scatter_kw.update({key:val})

            else: 
                raise ValueError(f'Only floats allowed, found {", ".join(self.dtypes_in)}')

        else:
            raise ValueError('Need a pandas DataFrame')



        if highlight:
            self.highlight = ((row[1], row[2], row.Index) for row in self.data_in.itertuples() if row.Index in highlight)
       


    def __repr__(self) -> str:
        return (
            f"XYview(X: {self.xlabel}, Y: {self.ylabel}\n"
            f"Observations: {len(self.data_in)}, Pearson r: {self.pearson:.2f})"
            )
    
    def _get_pearson_label(self, pearson_r):

        ptch = Patch(color='w')
        lbl = f'r: {pearson_r:.2f}'

        return [[ptch], [lbl]]

    def _get_reg_line(self, slope, intercept):

        line_props = {'lw':0.5, 'ls':':', 'color':'.15', 'zorder':-1}

        return Line2D(self.x, self.x*self.slope + self.intercept, **line_props)

