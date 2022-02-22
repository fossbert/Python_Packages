# computing
from collections import namedtuple
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
from .utils import DataDot, DataNum
from itertools import combinations



class Corrplot(Dotplot):
    
    def __init__(self, ) -> None:
        super().__init__()





class Dotplot:
    
    def __init__(self, data, 
                 x:str, y:str, 
                 size:str, 
                 color:str=None, 
                 fillna:tuple = None) -> None:
                
        self.data = DataDot(data, x=x, y=y, size=size, color=color)
        self.x, self.y = self._get_scatter_pos(self.data.ncols, self.data.nrows)
        self.size_fill, self.color_fill = self._check_fillna(fillna)
        self.tick_names, self.size_raw = self._piv_and_ravel(self.data.var_names.size, self.size_fill, True)
        if color:
            self.color_raw =  self._piv_and_ravel(self.data.var_names.color, self.size_fill)
        
    def _check_fillna(self, fillna:tuple):
        
        if not fillna:
            return (None, None)
    
        try:
            fillna = [float(x) for x in fillna]
        except ValueError:
            print("Could not convert to float, please reconsider!")
            raise
        else:
            if len(fillna)==1:
                print('Assuming you only want to provide fillna for size, setting color to None')
                return (fillna, None)
            else:
                return fillna
        
    def _get_scatter_pos(self, ncols:int, nrows:int):
        
        return [arr.ravel() for arr in np.meshgrid(np.arange(ncols), np.arange(nrows))]
    
    def _piv_and_ravel(self, values:str, fillna:float, get_tick_labels: bool=False):
        
        df_piv = self.data.df.pivot(index=self.data.var_names.y, columns=self.data.var_names.x, values=values)
        
        if fillna:
            df_piv = df_piv.fillna(fillna)
        
        valus_flat = df_piv.values.ravel()
        
        if get_tick_labels:
            tick_names = TickNames(df_piv.columns.to_list(), df_piv.index.to_list())
            return tick_names, valus_flat
        else:
            return valus_flat
            
        
    
    def cut_size(self, 
                 reverse: bool = False,
                 bins:tuple=None, 
                 sizes_out:tuple=None, 
                 transform:callable=None):
        
        sizes_raw = self.size_raw

        if not bins:
            bins = np.nanquantile(sizes_raw, q=[0, 1/3, 2/3, 1])
            
        if not sizes_out:
            sizes_out = np.arange(len(bins)-1)*20
            
        if reverse:
            sizes_out = sizes_out[::-1]
            
        if transform:
            sizes_raw, bins = [transform(i) for i in (sizes_raw, bins)]
        
        if len(bins)!=len(sizes_out)+1:
            raise ValueError(f'{len(bins)} bins, but {len(sizes_out)} sizes. That does not work.')
        
        self.size_cut = pd.cut(sizes_raw, bins, include_lowest=True, labels=sizes_out)
        self.bins = [(bin, size) for bin, size in zip(bins[1:], sizes_out)]
        
        print('Assigned sizes as follows:')
        [print(f'{bin:.2e} -> size {size}') for bin, size in self.bins]
            
    def set_ticklabels(self, ax=None, **text_kwargs):
        
        if ax is None:
            ax = plt.gca()
            
        ax.set_xticks(np.arange(self.data.ncols), self.tick_names.xticks, **text_kwargs)
        ax.set_yticks(np.arange(self.data.nrows), self.tick_names.yticks, **text_kwargs)
        
        
    def get_size_handles(self, reverse:bool=False, marker_sizes:tuple=None, num_fmt:str='1.1e', **line_kwargs):
        
        lbls = [f'<= {bin:{num_fmt}}' for bin, size in self.bins if size>0]
        
        if not marker_sizes:
            marker_sizes = np.arange(1, len(lbls)+1)*4
        
        if len(lbls)!=len(marker_sizes):
            raise ValueError(f'{len(lbls)} bins, but {len(marker_sizes)} sizes. That does not work.')
        
        if reverse:
            marker_sizes = marker_sizes[::-1]
        
        return [Line2D([0], [0], marker='o', color='w', markerfacecolor='0.25', label=l, markersize=s) for l, s in zip(lbls, marker_sizes)]





TickNames = namedtuple('TickNames', 'xticks yticks')
        