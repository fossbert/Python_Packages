# pillars of work
from sqlite3 import connect
import numpy as np
import pandas as pd

# tools and stats
from itertools import combinations
from collections import namedtuple

# Plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex

# Utils
from .utils import DataMix, DataNum, _cut_p

# stats
from scipy.stats import (kruskal, f_oneway, ttest_ind, mannwhitneyu, wilcoxon)
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

# documentation 
from typing import Union, Sequence


class StripBox:
    """
    A class to hold and process numeric data across one categorical variable with at least two levels,
    and provide statistical analysis and visualization tools such as boxplots, strip plots, and p-value annotations.
    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame containing the numeric variable and one categorical variable.
    jitter : float, optional
        Amount of jitter to apply to strip plot points (default is 0.08).
    p_method_global : str, optional
        Method for calculating the global p-value. Options are 'Kruskal' or 'Anova' (default is 'Kruskal').
    s1_order : list, optional
        Custom order for the categorical variable levels. Must match existing categories.
    s1_colors : tuple, optional
        Colors for each category. Must match the number of categories.
    Attributes
    ----------
    data : DataMix
        Internal data wrapper for input DataFrame.
    ylabel : str
        Name of the numeric variable.
    s1 : str
        Name of the categorical variable.
    s1_dtype : str
        Data type of the categorical variable.
    jitter : float
        Jitter value for strip plot.
    s1_categories : list
        List of category levels.
    s1_colors : list
        List of colors for each category.
    p_method_global : str
        Method used for global p-value calculation.
    global_p : float
        Calculated global p-value.
    strip_data : pd.DataFrame
        Data prepared for strip plot visualization.
    pairwise_stats : pd.DataFrame
        DataFrame containing pairwise statistical test results (set after calling `calc_pairwise_p`).
    Methods
    -------
    __repr__():
        String representation of the StripBox instance.
    _calc_global_p(method):
        Calculate the global p-value using the specified method.
    calc_pairwise_p(posthoc_method='dunn'):
        Calculate pairwise p-values between groups using posthoc tests ('dunn' or 'tukey').
    boxplt(ax=None, adjust_x=False, **boxplot_kwargs):
        Plot a boxplot of the numeric variable split by the categorical variable.
    add_strips(ax=None, **scatter_kwargs):
        Add strip plot points to the current axes.
    add_global_p(ax=None, **legend_kwargs):
        Add the global p-value as a legend to the plot.
    add_pair_p(groupA, groupB, yoffset=0.2, connect_top=True, cut_p=False, ax=None, line_kwargs=None, **text_kwargs):
        Annotate the plot with pairwise p-value between two groups.
    Raises
    ------
    ValueError
        If input data does not meet requirements, or if provided category order/colors do not match available categories.

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
        """
        Calculates the global p-value for comparing groups in the dataset using the specified statistical method.
        Parameters
        ----------
        method : str
            The statistical test to use for global comparison. Must be one of 'Kruskal' or 'Anova'.
        Returns
        -------
        float
            The computed global p-value. Returns np.nan if only one group is present.
        Raises
        ------
        ValueError
            If an invalid method is provided.
        Notes
        -----
        - For two groups: uses Mann-Whitney U test ('Kruskal') or Welch's t-test ('Anova').
        - For more than two groups: uses Kruskal-Wallis test ('Kruskal') or one-way ANOVA ('Anova').
        - If only one group is present, no p-value is calculated.
        """
        
        options = ['Kruskal', 'Anova']
        
        if method not in options:
            raise ValueError(f'Valid global value methods are: {", ".join(options)}, got: {method}')

        alternatives = {'Kruskal':'MWU', 'Anova':"Welch"}
        
        bp_arrays = _get_bp_arrays(self.data.df, self.ylabel, self.s1)
        
        if len(self.s1_categories)==1:
            print('No p-value for single group!')
            return np.nan
        elif len(self.s1_categories)<3:
            _, pval = mannwhitneyu(*bp_arrays) if method=='Kruskal' else ttest_ind(*bp_arrays, equal_var=False)
            self.p_method_global = alternatives.get(method)
        else:
            _, pval = kruskal(*bp_arrays) if method == 'Kruskal' else f_oneway(*bp_arrays)
            
        return pval
    
    def calc_pairwise_p(self, posthoc_method:str='dunn'):
        """
        Calculates pairwise posthoc p-values between groups for boxplot visualization.
        Parameters
        ----------
        posthoc_method : str, optional
            Method for posthoc pairwise comparison. Supported options are 'dunn' (default) and 'tukey'.
        Raises
        ------
        ValueError
            If less than 3 groups are present or if an invalid posthoc method is specified.
        Updates
        -------
        self.pairwise_stats : pd.DataFrame
            DataFrame containing pairwise group comparisons, including x positions, min/max values, and adjusted p-values.
        """

        
        bp_arrays = _get_bp_arrays(self.data.df, self.ylabel, self.s1)
        
        if (ngroups:=len(bp_arrays))<3:
            raise ValueError(f'No need for posthoc adjustment with {ngroups} groups!')

        box_max_values, box_min_values = list(zip(*[(np.max(i), np.min(i)) for i in bp_arrays]))
         
        posthoc_options = ['dunn', 'tukey']
        
        if posthoc_method not in posthoc_options:
            raise ValueError(f'Valid global value methods are: {", ".join(posthoc_options)}, got: {posthoc_method}')
        
        if posthoc_method=='dunn':
            posthoc = sp.posthoc_dunn(self.data.df, val_col=self.ylabel, group_col=self.s1, p_adjust="fdr_bh")
        else:
            posthoc = sp.posthoc_tukey(self.data.df, val_col=self.ylabel, group_col=self.s1)

        combos = combinations(range(len(posthoc.columns)), 2)

        index = []
        data = []
        for i, j in combos:
            
            contrast = (posthoc.columns[i], posthoc.columns[j])
            xpos = (i+1, j+1)
            ymax = (box_max_values[i], box_max_values[j])
            ymin = (box_min_values[i], box_min_values[j])
            
            index.append(contrast)
            data.append((xpos, ymax, ymin, posthoc.iloc[i, j]))
            
        index_mult = pd.MultiIndex.from_tuples(index, names=["GroupA", "GroupB"])
        df = pd.DataFrame(data,  index=index_mult, columns=['xpos', 'ymax', 'ymin', 'padj'])
        
        self.pairwise_stats = df
        
    
    def boxplt(self, ax=None, adjust_x=False, **boxplot_kwargs):
        """
        Plots a boxplot for the data contained in the object.
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the boxplot. If None, uses current axes.
        adjust_x : bool, default False
            If True, adjusts the x-tick labels for better readability.
        **boxplot_kwargs
            Additional keyword arguments passed to `ax.boxplot`.
        Returns
        -------
        None
        """
        
        if ax is None:
            ax = plt.gca()

        x_tick_props = {}

        bp_arrays = _get_bp_arrays(self.data.df, self.ylabel, self.s1)
        
        ax.boxplot(bp_arrays, **boxplot_kwargs)

        if adjust_x:
            x_tick_props.update({"rotation":60, "rotation_mode":"anchor", "ha":"right"})
            
        ax.set_xticks(np.arange(len(self.s1_categories))+1, self.s1_categories, **x_tick_props)
        
    def add_strips(self, ax=None, **scatter_kwargs):
        """
        Adds strip (scatter) plot to the given axes using the data in `self.strip_data`.
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, uses current axes.
        **scatter_kwargs
            Additional keyword arguments passed to `ax.scatter`.
        Returns
        -------
        None
        """
        
        if ax is None:
            ax = plt.gca()
            
        scatter_call = dict(zip(self.strip_data.columns, self.strip_data.T.values))
        
        if scatter_kwargs:
            for k, v in scatter_kwargs.items():
                scatter_call.update({k:v})
                
        ax.scatter(**scatter_call)
        
    def add_global_p(self, ax=None, **legend_kwargs):
        """
        Adds a legend to the given axis displaying the global p-value.
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axis to which the legend will be added. If None, uses current axis.
        **legend_kwargs : dict
            Additional keyword arguments passed to `ax.legend()`.
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axis with the added legend.
        """
        
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
        
        """
        Adds a pairwise p-value annotation between two groups on a boxplot.
        Parameters
        ----------
        groupA : str
            Name of the first group.
        groupB : str
            Name of the second group.
        yoffset : float, optional
            Vertical offset for the annotation line and text (default: 0.2).
        connect_top : bool, optional
            If True, connects annotation line to the top of the boxes; otherwise to the bottom (default: True).
        cut_p : bool, optional
            If True, formats p-value using _cut_p; otherwise uses scientific notation (default: False).
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, uses current axes.
        line_kwargs : dict, optional
            Additional keyword arguments for the annotation line.
        **text_kwargs
            Additional keyword arguments for the annotation text.
        Raises
        ------
        AttributeError
            If the pairwise statistics DataFrame is not found.
        KeyError
            If groupA and/or groupB are not present in the statistics DataFrame.
        """

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
    """
    A class for creating split boxplots with overlaid strip plots for two categorical variables.
    This class organizes data into two categorical splits and visualizes the numeric variable
    using boxplots and strip plots, with customizable spacing, jitter, and colors.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing one numeric variable and two categorical variables.
    minsize : int, default=3
        Minimum number of data points required per group.
    jitter : float, default=0.08
        Amount of horizontal jitter applied to strip points.
    bp_width : float, default=0.5
        Width of each boxplot.
    space_within : float, default=0.1
        Space between boxplots within the same split.
    space_between : float, default=0.5
        Space between groups of boxplots (between split 1 levels).
    s1_order : list, optional
        Custom order for the first split's categorical levels.
    s2_order : list, optional
        Custom order for the second split's categorical levels.
    strip_colors : list, optional
        List of colors for strip points corresponding to split 2 levels.
    Attributes
    ----------
    data : DataMix
        Processed data object with categorical splits.
    ylabel : str
        Name of the numeric variable.
    s1 : str
        Name of the first categorical split variable.
    s2 : str
        Name of the second categorical split variable.
    s1_categories : list
        Ordered levels for split 1.
    s2_categories : list
        Ordered levels for split 2.
    bp_width : float
        Width of each boxplot.
    space_within : float
        Space between boxplots within a split.
    space_between : float
        Space between split 1 groups.
    jitter : float
        Jitter for strip points.
    strip_colors : list
        Colors for strip points.
    n_s1 : int
        Number of split 1 levels.
    n_s2 : int
        Number of split 2 levels.
    s1_ticks : TickInfo
        Tick information for split 1.
    xtick_pos : list
        Positions for all boxplots.
    strip_data : pd.DataFrame
        Data for strip plot points.
    handles : list
        Legend handles for strip colors.
    Methods
    -------
    __repr__():
        String representation of the SplitStripBox object.
    _get_xgrid():
        Computes grid positions for boxplots and tick labels.
    boxplt(ax=None, adjust_x=False, **boxplot_kwargs):
        Plots boxplots on the given axis.
    add_strips(ax=None, **scatter_kwargs):
        Adds strip plot points to the given axis.
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
        self.strip_data = _add_strip_data(self.data.df, 
                                          self.ylabel, 
                                          self.xtick_pos,
                                          self.jitter, 
                                          self.strip_colors, 
                                          self.s1,
                                          self.s2)
        # Convenience for adding a legend
        self.handles = [Line2D([0], [0], color='w', marker='o', markerfacecolor=c, label=l) for c, l in zip(self.strip_colors, self.s2_categories)]
            
    def __repr__(self) -> str:
        return (
            f"SplitStripBox(Numeric variable: {self.ylabel}\n"
            f"Split 1: {self.s1}, levels: {len(self.s1_categories)}\n"
            f"Split 2: {self.s2}, levels: {len(self.s2_categories)}\n "
            f"Total data points: {len(self.data.df)})"
        )

    def _get_xgrid(self):
        
        unit_width = self.bp_width/2 + self.space_within/2 # half of one single unit (boxplot plus space)
        s2_width = self.bp_width*self.n_s2 + (self.n_s2-1)*self.space_within #width of all s2 boxplots plus space

        total_width = s2_width + self.space_between

        s1_grid = np.arange(0, round(self.n_s1*total_width, 1), round(total_width, 1))

        if len(s1_grid) > self.n_s1:
            s1_grid = s1_grid[:self.n_s1] # there are some instances (probably due to rounding where the above returns a grid of n_s1+1)

        # compute a grid of s2 level boxplot positions for every s1 level
        xpos = [np.linspace(-unit_width*(self.n_s2-1), unit_width*(self.n_s2-1), self.n_s2) + i for i in s1_grid]
        s1_xtick_pos = [round(np.median(x), 1) for x in xpos] # median for xtick labeling
        all_xtick_pos = [round(y, 1) for x in xpos for y in x] # to pass to ax.boxplot
        
        return TickInfo(s1_xtick_pos, self.s1_categories), all_xtick_pos
        
    def boxplt(self, ax=None, adjust_x=False, **boxplot_kwargs):
    
        if ax is None:
            ax = plt.gca()
        
        x_tick_props = {}

        bp_arrays = _get_bp_arrays(self.data.df, self.ylabel, self.s1, self.s2)
        
        ax.boxplot(bp_arrays, widths=self.bp_width, positions=self.xtick_pos, **boxplot_kwargs)

        if adjust_x:
            x_tick_props.update({"rotation":60, "rotation_mode":"anchor", "ha":"right"})
     
        ax.set_xticks(*self.s1_ticks, **x_tick_props)

    def add_strips(self, ax=None, **scatter_kwargs):
        
        if ax is None:
            ax = plt.gca()
            
        scatter_call = dict(zip(self.strip_data.columns, self.strip_data.T.values))
        
        if scatter_kwargs:
            for k, v in scatter_kwargs.items():
                scatter_call.update({k:v})
                
        ax.scatter(**scatter_call)


class PairedStripBox:
    """
    PairedStripBox(data: pd.DataFrame, jitter: float = 0.08)
    Visualizes paired data using boxplots and strip plots, with options to connect paired points and annotate statistical significance.
    
    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing paired numerical data. Each column represents a group/variable.
    jitter : float, default=0.08
        Amount of horizontal jitter applied to strip points for visualization.
    
    Attributes
    ----------
    data : DataNum
        Internal representation of the input data.
    jitter : float
        Jitter applied to strip points.
    xtick_pos : np.ndarray
        Positions of x-ticks for plotting.
    xtick_names : list
        Names of variables/groups for x-ticks.
    Y : np.ndarray
        Cleaned (dropna) data values.
    X : np.ndarray
        Jittered x-positions for strip points.
    stat_df : pd.DataFrame
        DataFrame containing pairwise statistical test results.
    
    Methods
    -------
    __repr__() -> str
        Returns a string representation of the object.
    boxplt(ax=None, adjust_x=False, **boxplot_kwargs)
        Plots boxplots for each group/variable.
    add_strips(strip_colors=None, ax=None, **scatter_kwargs)
        Adds strip points to the plot, colored by group.
    connect_strips(ax=None, **line_kwargs)
        Connects paired data points across groups.
    add_pair_p(group_A, group_B, connect_top=True, yoffset=0.2, cut_pval=True, ax=None, line_kwargs=None, **text_kwargs)
        Annotates the plot with pairwise p-values between two groups.
    _calc_pvals()
        Calculates pairwise p-values using Wilcoxon Signed Rank test, with Bonferroni correction for multiple comparisons.
    """
    

    def __init__(self, 
                 data:pd.DataFrame,     
                 jitter:float=0.08) -> None:
        
        self.data = DataNum(data)
        self.jitter = jitter
        self.xtick_pos, self.xtick_names = np.arange(1, self.data.ncols+1), self.data.var_names
        self.Y = self.data.df.dropna().values
        self.X = np.random.normal(loc=self.xtick_pos, scale=self.jitter, size=self.Y.shape)
        self.stat_df = self._calc_pvals()

       # Once we made it past all these checks, we can start with some calculations   

    def __repr__(self) -> str:
        return (
            f"PairedStripBox(Num of boxplots: {self.data.ncols}\n"
            f"Total data points: {len(self.data.df)})"
        )
        
    def boxplt(self, ax=None, adjust_x=False, **boxplot_kwargs):
    
        if ax is None:
            ax = plt.gca()

        x_tick_props = {}
        
        bp_props = {'widths':0.5}

        if boxplot_kwargs:
            for k,v in boxplot_kwargs.items():
                bp_props.update({k:v})
        
        ax.boxplot(self.Y, **bp_props)
        
        if adjust_x:
            x_tick_props.update({"rotation":60, "rotation_mode":"anchor", "ha":"right"})
            
        ax.set_xticks(self.xtick_pos, self.xtick_names, **x_tick_props)

    def add_strips(self, strip_colors:Sequence=None, ax=None, **scatter_kwargs):

        if strip_colors is None: 
            strip_colors = [to_hex(hue) for hue in plt.get_cmap('Blues')(np.linspace(0.5, 1, self.data.ncols))]
        else:
            strip_colors = list(strip_colors)
            if len(strip_colors) != self.data.ncols:
                raise ValueError(f"Need {self.data.ncols} colors, got {len(strip_colors)}")

        if ax is None:
            ax = plt.gca()

        scatter_call = {'x':self.X, 'y':self.Y, 's':len(self.data.df)/25, 'c':np.tile(strip_colors, len(self.Y))}
        
        if scatter_kwargs:
            for k, v in scatter_kwargs.items():
                scatter_call.update({k:v})
                
        ax.scatter(**scatter_call)


    def connect_strips(self, ax=None, **line_kwargs):

        if ax is None:
           ax = plt.gca()

        line_props = {'lw':0.25, 'c':'0.5', 'alpha':0.7, 'zorder':-1}

        if line_kwargs:
            for k, v in line_kwargs.items():
                line_props.update({k:v})

        ax.plot(self.X.T, self.Y.T, **line_props)


    def add_pair_p(self, 
                   group_A:str, 
                   group_B:str, 
                   connect_top: bool = True, 
                   yoffset:float=0.2,
                   cut_pval:bool = True,
                   ax=None,
                   line_kwargs:dict = None,
                   **text_kwargs):
        """
        Annotates a boxplot with a p-value between two groups by drawing a connecting line and displaying the p-value.
        Parameters
        ----------
        group_A : str
            The name of the first group to compare.
        group_B : str
            The name of the second group to compare.
        connect_top : bool, optional
            If True, connects the top of the boxes; otherwise, connects the bottom. Default is True.
        yoffset : float, optional
            Vertical offset for the connecting line. Default is 0.2.
        cut_pval : bool, optional
            If True, formats the p-value using a custom function; otherwise, displays the raw p-value. Default is True.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot. If None, uses the current axes.
        line_kwargs : dict, optional
            Additional keyword arguments for customizing the connecting line.
        **text_kwargs
            Additional keyword arguments for customizing the p-value text annotation.
        Returns
        -------
        None
            This function modifies the plot in-place and does not return anything.
        Notes
        -----
        - Uses `self.stat_df` for statistical results and `self.Y` for boxplot data.
        - The function assumes that the statistical results and boxplot data are properly formatted.
        """

        line_props = {'lw':0.5, 'c':'0.15'}

        if line_kwargs:
            line_props.update(line_kwargs)
            
        text_props = {'fontsize':"x-small", 'ha':'center'}
        
        if text_kwargs:
            for k, v in text_kwargs.items():
                text_props.update({k:v})
     
        stats = self.stat_df.copy()

        xpos = np.array(stats.loc[(group_A, group_B), 'xpos'])

        line_xpos = np.repeat(xpos, 2) + 1

        ypos = np.max(self.Y, axis=0)[xpos] if connect_top else np.min(self.Y, axis=0)[xpos]

        line_ypos = _get_connect_lines(ypos, yoffset, connect_top)

        if ax is None:
            ax = plt.gca()

        ax.plot(line_xpos, line_ypos, **line_props)

        xtext = np.sum(xpos+1)/2
        ytext = np.max(line_ypos) if connect_top else np.min(line_ypos)
        pval = "fwer" if stats.shape[1]>2 else "pval"
        pval = stats.loc[(group_A, group_B), pval]
        pval_string = _cut_p(pval) if cut_pval else f'{pval:.2e}'

        ax.text(xtext, ytext, pval_string, ha='center')


    def _calc_pvals(self):
        """
        Calculates pairwise Wilcoxon signed-rank test p-values between all columns in the dataset.
        For each unique pair of columns, computes the Wilcoxon test and stores the p-value,
        the variable names, and their positions. If three or more pairs are tested, applies
        Bonferroni correction for multiple hypothesis testing.
        
        Returns
        -------
        stats : pandas.DataFrame
            DataFrame containing pairwise positions ('xpos'), raw p-values ('pval'), and
            Bonferroni-adjusted p-values ('fwer', if applicable). The index consists of
            tuples of variable names for each pair.
        """

        

        pair_names = []
        pvals = []
        pair_xpos = []

        for i, j in combinations(range(self.data.ncols), 2):

            pair_names.append((self.data.var_names[i], self.data.var_names[j]))

            pvals.append(wilcoxon(self.Y.T[i], self.Y.T[j], zero_method="zsplit")[1])

            pair_xpos.append((i, j))


        stats = pd.DataFrame({'xpos':pair_xpos, 'pval':pvals}, index=pd.MultiIndex.from_tuples(pair_names))

        if len(stats)>=3:
            print('Adjusting for testing multiple hypotheses using the Bonferroni method.')
            stats['fwer'] = multipletests(stats['pval'], method='bonferroni')[1] 

        return stats

      


### Helper functions

def _get_bp_arrays(data: pd.DataFrame, value_var:str, *categories):
    
    categories = list(categories)

    return data.groupby(categories, observed=False).apply(lambda x: x[value_var].values).values


def _add_strip_data(data:pd.DataFrame, value_var:str, xpos:list, jitter:float, colors:list, *categories):
        
        categories = list(categories)

        if len(categories)==1:
            categories = categories.pop()

        grouped_df = data.groupby(categories, observed=False)

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