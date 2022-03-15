# pillars of work
import numpy as np
import pandas as pd

# tools and stats
from itertools import combinations
from collections import namedtuple

# Plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_hex

# Utils
from .utils import DataMix, _cut_p, categorical_cmap

# stats
from scipy.stats import (kruskal, f_oneway, ttest_ind, mannwhitneyu)
from statsmodels.stats.multitest import multipletests

# documentation 
from typing import Union




class StripBox:

    """[Class to hold and process numeric data across one categorical variable holding at least two ]
    """
    
    def __init__(self, 
                data:pd.DataFrame, 
                 jitter: float = 0.08,
                 p_method_global:str = 'Kruskal',
                 s1_order: list = None,
                 s1_colors:tuple = None) -> None:
        
        self.data = DataMix(data, ncat=1, minsize=1)
        self.ylabel, self.s1 = self.data.var_names
        _, self.s1_dtype = self.data.dtypes
        self.jitter = jitter
      
        # Fix order category if user wishes to do so
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
    
        if s1_colors:
            if len(s1_colors)!=len(self.s1_categories):
                raise ValueError("Provided colors did not match number of categories")
            self.s1_colors = [color if len(color)==1 else to_hex(color) for color in s1_colors]
        else: 
            self.s1_colors = ['C'+str(i) for i in range(len(self.s1_categories))]
        
        # global p-value does not care about order, so can calculate here
        self.p_method_global = p_method_global
        self.global_p = self._calc_global_p(self.p_method_global)
        
        # Scatter plot data
        self.strip_data = _add_strip_data(self.data.df, 
                                          self.ylabel, 
                                          list(range(1, len(self.s1_categories)+1)),
                                          self.jitter, 
                                          self.s1_colors, 
                                          self.s1)
       
            
    def __repr__(self) -> str:
        
        return (
            f"StripBox(Numeric variable: {self.ylabel}\n"
            f"Split 1: {self.s1}, levels: {len(self.s1_categories)}\n"
            f"Total data points: {len(self.data.df)})"
        )      
        
    def _calc_global_p(self, method:str):
        
        options = ['Kruskal', 'Anova']
        
        if method not in options:
            raise ValueError(f'Valid global value methods are: {", ".join(options)}, got: {method}')

        alternatives = {'Kruskal':'MWU', 'Anova':"Welch"}
        
        bp_arrays = _get_bp_arrays(self.data.df, self.ylabel, self.s1)
        
        if len(self.s1_categories)<3:
            _, pval = mannwhitneyu(*bp_arrays) if method=='Kruskal' else ttest_ind(*bp_arrays, equal_var=False)
            self.p_method_global = alternatives.get(method)
        else:
            _, pval = kruskal(*bp_arrays) if method == 'Kruskal' else f_oneway(*bp_arrays)
            
        return pval
    
    def calc_pairwise_p(self, pair_method:str='welch', adj_method:str='fdr_bh'):
        
        bp_arrays = _get_bp_arrays(self.data.df, self.ylabel, self.s1)
        box_max_values, box_min_values = list(zip(*[(np.max(i), np.min(i)) for i in bp_arrays]))
         
        pair_options = ['welch', 'mannwhitneyu']
        
        adj_options = ['fdr_bh', 'bonferroni']
        
        if pair_method not in pair_options:
            raise ValueError(f'Valid global value methods are: {", ".join(pair_options)}, got: {pair_method}')
        
        if adj_method not in adj_options:
            raise ValueError(f'Valid global value methods are: {", ".join(adj_options)}, got: {adj_method}')
        
        combos = combinations(range(len(self.s1_categories)), 2)

        index = []
        data = []
        for i, j in combos:
            
            contrast = (self.s1_categories[i], self.s1_categories[j])
            xpos = (i+1, j+1)
            ymax = (box_max_values[i], box_max_values[j])
            ymin = (box_min_values[i], box_min_values[j])
            
            if pair_method == 'Welch':
                _, pval = ttest_ind(bp_arrays[i], bp_arrays[j], equal_var=False)
            else:
                _, pval = mannwhitneyu(bp_arrays[i], bp_arrays[j])
    
            index.append(contrast)
            data.append((xpos, ymax, ymin, pval))
            
        index_mult = pd.MultiIndex.from_tuples(index, names=["GroupA", "GroupB"])
        df = pd.DataFrame(data,  index=index_mult, columns=['xpos', 'ymax', 'ymin', 'pval'])
        if len(df)>1:
            padj =  multipletests(df['pval'].values, method=adj_method)[1]
            df.insert(4, 'padj', padj)
        
        self.pairwise_stats = df
        
    
    def boxplt(self, ax=None, **boxplot_kwargs):
        
        if ax is None:
            ax = plt.gca()
        
        bp_arrays = _get_bp_arrays(self.data.df, self.ylabel, self.s1)
        
        ax.boxplot(bp_arrays, **boxplot_kwargs)
        ax.set_xticks(np.arange(len(self.s1_categories))+1, self.s1_categories)
        
    def add_strips(self, ax=None, **scatter_kwargs):
        
        if ax is None:
            ax = plt.gca()
            
        scatter_call = dict(zip(self.strip_data.columns, self.strip_data.T.values))
        
        if scatter_kwargs:
            for k, v in scatter_kwargs.items():
                scatter_call.update({k:v})
                
        ax.scatter(**scatter_call)
        
    def add_global_p(self, ax=None, **legend_kwargs):
        
        if ax is None:
            ax = plt.gca()
            
        def_kwargs = {'loc':0, 'handlelength':0, 'handletextpad':0, 
        "frameon":False, 'fontsize':'x-small', 'labelcolor':'0.15'}
        
        if legend_kwargs:
            for k,v in legend_kwargs.items():
                def_kwargs.update({k:v})
                
        ax.legend([Patch(color='w')], [f'{self.p_method_global} p: {self.global_p:1.2e}'], **def_kwargs)
        
        return ax


    def add_pair_p(self,
                   groupA:str, 
                   groupB:str, 
                   yoffset:float=0.2, 
                   connect_top: bool =True,
                   cut_p:bool=False, 
                   ax=None,line_kwargs:dict=None, **text_kwargs):
        
        if ax is None:
            ax = plt.gca()
        
        line_props = {'lw':0.5, 'c':'0.15'}
        if line_kwargs:
            line_props.update(line_kwargs)
            
        text_props = {'fontsize':"xx-small", 'ha':'center'}
        
        if text_kwargs:
            for k, v in text_kwargs.items():
                text_props.update({k:v})
            
        try:
            subseries = self.pairwise_stats.loc[groupA].loc[groupB]
            xpos = np.repeat(subseries.xpos, 2)
            xtext = np.sum(subseries.xpos)/2
            if connect_top:
                ypos = _get_connect_lines(subseries.ymax, yoffset)
                ytext = np.max(ypos)
            else:
                ypos = _get_connect_lines(subseries.ymin, yoffset, top=False)
                ytext = np.min(ypos)
    
            ax.plot(xpos, ypos, **line_props)
            pval = subseries.padj if 'padj' in subseries else subseries.pval
            pstring = _cut_p(pval) if cut_p else f'{pval:1.2e}'
            ax.text(xtext, ytext, pstring, **text_props)
        
        except AttributeError:
            print(f'Could not find stat DataFrame')
            raise
        except KeyError:
             print(f'Could not find {groupA} and/or {groupB} in stat DataFrame')
             raise



class SplitStripBox:
       
    """Container for producing a box- and strip plots where the x-axis is split by two levels, i.e. 
    an outer and an inner level.
    
    Parameters
    ----------
    data : pandas DataFrame
        DataFrame holding information on one numeric and two categorical/string variables.
    
    minsize : int
        Minimal number of samples per subgroup, defaults to 3.
    
    jitter : float
        Amount of noise to add to dots in strip chart, defaults to 0.08
        
    bp_width : float
        Width of boxplots, important for x-position calculation, defaults to 0.5
    
    space_within : float
        Space between individual boxplots of an s1 group, defaults to 0.1
    
    space_between : float 
        Space between s1 groups, defaults to 0.5
        
    s1_order: list 
        Option to reorder levels of outer split s1
    
    s2_order: list
        Option to reorder levels of inner split s2
    
    strip_colors: list 
        Option to provide s2 colors, i.e. for each boxplot within a s1 split. Defaults to Greys internally. 
        Number of provided colors must match the number of levels of s2. 
    
    """

    def __init__(self, 
                 data:pd.DataFrame,     
                minsize:int = 3,
                jitter:float=0.08,
                bp_width:float=0.5,
                space_within:float=0.1,
                space_between:float=0.5,    
                s1_order: list = None, 
                s2_order: list = None,
                strip_colors: list =None) -> None:
        
        self.data = DataMix(data, ncat=2, minsize=minsize)
        self.ylabel, self.s1, self.s2 = self.data.var_names
        _, self.s1_dtype, self.s2_dtype = self.data.dtypes
        self.bp_width = bp_width
        self.space_within = space_within
        self.space_between = space_between
        self.jitter = jitter
       
        if self.s1_dtype == 'string':
            self.data.df[self.s1] = self.data.df[self.s1].astype('category')
            self.s1_dtype = 'categorical'
            
        if self.s2_dtype == 'string':
            self.data.df[self.s2] = self.data.df[self.s2].astype('category')
            self.s2_dtype = 'categorical'
        
        self.s1_categories = self.data.df[self.s1].cat.categories.to_list()
        self.s2_categories = self.data.df[self.s2].cat.categories.to_list()
        
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
        
        # convenience for grid computation
        self.n_s1 = len(self.s1_categories)
        self.n_s2 = len(self.s2_categories)
    
        if strip_colors:
            if needed:=len(self.s2_categories)!=len(strip_colors):
                raise ValueError(f"Provided strip colors did not match number of {needed} s2 categories")
            scolors = np.tile(strip_colors, self.n_s1)
            self.strip_colors = [color if len(color)==1 else to_hex(color) for color in scolors]
        else: 
            # Default will use hues of grey
            grey_colors = plt.get_cmap('Greys')(np.linspace(0.3, 0.6, self.n_s2))
            self.strip_colors = [to_hex(color) for color in np.tile(grey_colors, (self.n_s1, 1))]
            
       # Once we made it past all these checks, we can start with some calculations   
        self.s1_ticks, self.xtick_pos = self._get_xgrid()
        # self.strip_data = 
        self.strip_data = _add_strip_data(self.data.df, 
                                          self.ylabel, 
                                          self.xtick_pos,
                                          self.jitter, 
                                          self.strip_colors, 
                                          self.s1,
                                          self.s2)
            
    def __repr__(self) -> str:
        return (
            f"SplitStripBox(Numeric variable: {self.ylabel}\n "
            f"Split 1: {self.s1}, levels: {len(self.s1_categories)}\n"
            f"Split 2: {self.s2}, levels: {len(self.s2_categories)}\n "
            f"Total data points: {len(self.data.df)})"
        )

    def _get_xgrid(self):
        
        unit_width = self.bp_width/2 + self.space_within/2 # half of one single unit (boxplot plus space)
        s2_width = self.bp_width*self.n_s2 + (self.n_s2-1)*self.space_within #width of all s2 boxplots plus space

        total_width = s2_width + self.space_between
        s1_grid = np.arange(0, self.n_s1*total_width, total_width)

        # compute a grid of s2 level boxplot positions for every s1 level
        xpos = [np.linspace(-unit_width*(self.n_s2-1), unit_width*(self.n_s2-1), self.n_s2) + i for i in s1_grid]
        s1_xtick_pos = [np.median(x) for x in xpos] # median for xtick labeling
        all_xtick_pos = [y for x in xpos for y in x] # to pass to ax.boxplot
        
        return TickInfo(s1_xtick_pos, self.s1_categories), all_xtick_pos
        
    def boxplt(self, ax=None, **boxplot_kwargs):
    
        if ax is None:
            ax = plt.gca()
        
        bp_arrays = _get_bp_arrays(self.data.df, self.ylabel, self.s1, self.s2)
        
        ax.boxplot(bp_arrays, widths=self.bp_width, positions=self.xtick_pos, **boxplot_kwargs)
        ax.set_xticks(*self.s1_ticks)

    def add_strips(self, ax=None, **scatter_kwargs):
        
        if ax is None:
            ax = plt.gca()
            
        scatter_call = dict(zip(self.strip_data.columns, self.strip_data.T.values))
        
        if scatter_kwargs:
            for k, v in scatter_kwargs.items():
                scatter_call.update({k:v})
                
        ax.scatter(**scatter_call)


### Helper functions

def _get_bp_arrays(data: pd.DataFrame, value_var:str, *categories):
    
    return data.groupby(list(categories)).apply(lambda x: x[value_var].values).values


def _add_strip_data(data:pd.DataFrame, value_var:str, xpos:list, jitter:float, colors:list, *categories):
        
        grouped_df = data.groupby(list(categories))

        res = []
        
        for i, (_, subdf), c in zip(xpos, grouped_df, colors):
            
            res.append(_get_jitter_and_colors(subdf, x=value_var, loc=i, jitter=jitter, color=c))

        return pd.concat(res)



def _get_jitter_and_colors(data:pd.DataFrame, x:str, jitter:float, loc:int, color:str):
    """[This function is the main work horse to go from a series of numeric (y-) values to 
    a data necessary to plot jittered dots.]

    Returns:
        [pd.DataFrame]: [DataFrame with x and y values for a jittered strip chart.]"""
        
    yvals = data[x].values
    xvals = np.random.normal(loc=loc, scale=jitter, size=len(yvals))
    color_vec = np.repeat(color, len(yvals))
    
    return pd.DataFrame({'x':xvals, 'y':yvals, 'c':color_vec})


def _get_connect_lines(ys, offset, top=True):

    """[Helper function that when given two values will generate a list of 4 y values to define a connection line
    between the two points.]

    Returns:
        [list]: [containing a numeric y values for a connection line between two boxplots]
    """

    off_mult = (1, 2, 2, 1)

    if top:
        ycourse = np.repeat(ys, (3,1)) if np.argmax(ys)==0 else np.repeat(ys, (1,3))
        return [y+offset*i for y, i in zip(ycourse, off_mult)]  
    else:
        ycourse = np.repeat(ys, (3,1)) if np.argmin(ys)==0 else np.repeat(ys, (1,3))
        return [y-offset*i for y, i in zip(ycourse, off_mult)]
    
    
TickInfo = namedtuple('TickInfo', 'positions labels')