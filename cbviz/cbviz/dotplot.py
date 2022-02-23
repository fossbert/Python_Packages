# computing
from collections import namedtuple
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from adjustText import adjust_text

# stats
from scipy.spatial import distance
from scipy.cluster import hierarchy

# utils
from .utils import DataDot, DataNum, _color_light_or_dark


class Dotplot:
    
    """Class to compute and prepare data for a dotplot, that is a form of scatter plot where dots 
       are arranged in a fixed grid according to categorical variables on the x and y axis. 
       Two characterstics are generally used to convey information: size and color of each dot. 
       The numeric values underlying the scatter can be added as text. 
    """
    
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
    
    
    def __repr__(self) -> str:
        return repr(self.data)
    
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
        
        self.size_cut = pd.cut(sizes_raw, bins, include_lowest=True, labels=sizes_out, ordered=False) # allow duplicate values
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
    
    def annotate(self, scatter:PathCollection, ax=None, str_fmt:str="1.1f", **text_kwargs):
        
        if ax is None:
            ax = plt.gca()
            
        text_props = {"fontsize":'xx-small', "ha":'center',  'va':'center'}
        
        if text_kwargs:
            for k,v in text_kwargs.items():
                text_props.update({k:v})
        
        for x, y, t, s in zip(self.x, self.y, self.size_raw, self.size_cut):
            if not np.isnan(t) and s>0:
                color = _color_light_or_dark(np.array(scatter.to_rgba(t)))
                ax.text(y, x, f'{t:{str_fmt}}', color=color, **text_props)
        


### 

class Corrplot:
    
    """Convenience function which does the work to show a correlation plot of numeric variables in the 
    form of a Dotplot. A valid correlation method for panda's DataFrame.corr() needs to be passed as 
    correlation method. By default, complete hierarchical clustering of variables based on Spearman's correlation 
    distance is carried out. Options include adding in the diagonal (correlation of 1) and going from
    the upper left corner to the lower right. 
    """
    
    def __init__(self, data:pd.DataFrame,
                 corr_method:str='spearman', 
                 cluster: bool=True, 
                 upper_left:bool=True,
                 include_diagonal: bool=False
                 ) -> None:
        
        
        self.data = DataNum(data) # we need numeric variables for correlation
        self.corr_method = corr_method 
        self.cluster = cluster
        self.upper_left = upper_left
        self.include_diagonal = include_diagonal
        self.corr_df = self._corr_and_cluster(self.data.df)
        self.dp = Dotplot(data=self._mask_and_melt(self.corr_df), x='index', y='column', size='corr')
    
    
    def _mask_and_melt(self, corr_df:pd.DataFrame):
        
        # Masking includes the diagonal
        mask = np.ones(corr_df.shape).astype(bool)
        
        if self.include_diagonal:
            k_mask = 0
        elif self.upper_left:
            k_mask = -1
        else: 
            k_mask = 1
        corr_df = corr_df.where(np.tril(mask, k=k_mask)) if self.upper_left else corr_df.where(np.triu(mask, k=k_mask))
        
        # Melting
        corr_df_melt = corr_df.melt(var_name='column', value_name='corr', ignore_index=False).reset_index()
        # Fix order
        corr_df_melt['index'] = corr_df_melt['index'].astype('category').cat.reorder_categories(corr_df.columns.to_list())
        corr_df_melt['column'] = corr_df_melt['column'].astype('category').cat.reorder_categories(corr_df.columns.to_list())
        
        return corr_df_melt
    
    def _corr_and_cluster(self, data: pd.DataFrame):

        corr_df = data.corr(method=self.corr_method) # DataFrame correlation 
        
        if self.cluster:
            hlcust = hierarchy.linkage(distance.squareform(1 - corr_df), method='complete') # no option for Hierarch. Clustering
            ord = hierarchy.dendrogram(hlcust, labels=corr_df.index, no_plot=True)['leaves']
            corr_df = corr_df.iloc[ord, ord]
            
        return corr_df

### ------------------------------------------------------
TickNames = namedtuple('TickNames', 'xticks yticks')
        