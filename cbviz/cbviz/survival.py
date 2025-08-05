# pillars of work
import numpy as np
import pandas as pd

# tools
from itertools import combinations

# Plotting
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_hex

# stats
from scipy.stats import false_discovery_control as fdr


# documentation 
from typing import Union

# survival stuff
from sksurv.nonparametric import kaplan_meier_estimator as kme
from sksurv.compare import compare_survival


from .utils import DataSurv, _cut_p, _color_light_or_dark


class KME:
    """
    Convenience class for Kaplan-Meier survival analysis and visualization.
    The `KME` class provides a high-level interface for handling survival data, performing logrank tests,
    and plotting Kaplan-Meier survival curves, including stratification by categorical features and
    annotation of quantiles. It is designed to streamline survival analysis workflows and visualization
    in Python.

    Attributes
    data : DataSurv
        An instance of DataSurv containing a cleaned and preprocessed DataFrame with survival data.
    s1 : str
        Name of the annotating (grouping) feature used for stratification.
    s1_dtype : str
        Data type of the annotating feature ('string' or 'categorical'). Inferred from the DataFrame.
    csq : float
        Chi-square statistic from a logrank test comparing survival across groups.
    pval : float
        P-value from the logrank test.
    s1_categories : list of str
        List of unique categories/levels of the annotating feature.
    s1_colors : list of str
        List of color codes (strings or hex) for each category, either provided or auto-generated.
    handles : list of matplotlib.lines.Line2D
        List of legend handles for matplotlib, mapping colors to categories.
    pairwise_logranks : pd.DataFrame, optional
        DataFrame containing results of pairwise logrank tests (created by `calc_pairwise_logranks`).

    Methods
    __init__(data, s1_order=None, s1_colors=None)
        Initialize the KME object with survival data, optional category order, and colors.
    _label_quantile(x, xindx, lbl, q, label_offset, ax)
        Annotate a quantile value on a Kaplan-Meier plot.
    _intersect_km_curve(y, q)
        Find indices where the Kaplan-Meier curve crosses a specified quantile.
    _get_surv(stratify=False)
        Retrieve survival data, optionally stratified by the annotating feature.
    _plot_curve(surv, line_color, q, label_offset, ax, draw_ci, **step_kwargs)
        Plot a Kaplan-Meier curve with optional confidence intervals and quantile annotation.
    kmplot(stratify=False, draw_ci=True, q=0.5, label_offset=(2, 0.2), ax=None, **step_kwargs)
        Plot Kaplan-Meier survival curves, optionally stratified by group.
    add_global_p(cut_p=True, ax=None, **legend_kwargs)
        Add a global logrank p-value annotation to a matplotlib Axes.
    calc_pairwise_logranks()
        Compute pairwise logrank tests for all category combinations and store results.
    - This class assumes the existence of a `DataSurv` class for preprocessing and structuring survival data.
    - The plotting methods require matplotlib.
    - The class is intended for use in exploratory survival analysis and publication-quality visualization.
    """

    def __init__(self, 
                 data:pd.DataFrame,
                 s1_order: list = None,
                 s1_colors:list = None
                ) -> None:
        
        self.data = DataSurv(data)
        self.s1 = self.data.groups[0]
        *_, self.s1_dtype = self.data.dtypes
        self.csq, self.pval = compare_survival(self.data.surv, self.data.df[self.s1])
        self.time_max = self.data.df[self.data.time].max()
        
         # Fix order category if user wishes to do so
        if self.s1_dtype == 'string':
            self.data.df[self.s1] = self.data.df[self.s1].astype('category')
            self.s1_dtype = 'categorical'

        self.s1_categories = self.data.df[self.s1].cat.categories.to_list()
        
        if len(self.s1_categories) < 1:
            raise ValueError(f'There is not a single level for {self.s1}') 
        
        if s1_order:
             s1_order = list(s1_order)
             if all([s1_lvl in self.s1_categories for s1_lvl in s1_order]):
                 self.data.df[self.s1] = self.data.df[self.s1].cat.reorder_categories(s1_order)
                 self.s1_categories = s1_order
             else:
                 raise ValueError(f'Could not align levels, provided: {", ".join(s1_order)}, available: {", ".join(self.s1_categories)}')
    
        if s1_colors:
            s1_colors = list(s1_colors)
            if len(s1_colors)!=len(self.s1_categories):
                raise ValueError("Provided colors did not match number of categories")
            self.s1_colors = [color if len(color)==1 else to_hex(color) for color in s1_colors]
        else: 
            self.s1_colors = ['C'+str(i) for i in range(len(self.s1_categories))]

         # Convenience for adding a legend
        self.handles = [Line2D([0], [0], color=c, label=l, lw=3) for c, l in zip(self.s1_colors, self.s1_categories)]    

    def _label_quantile(self, x, xindx, lbl, q, label_offset, ax):
        """
        Annotate a quantile value on a Kaplan-Meier plot.

        Parameters
        ----------
        x : array-like
            The array of x-values (time points).
        xindx : int or array-like
            The index/indices in `x` where the quantile should be labeled.
        lbl : str
            The label text to annotate at the quantile.
        q : float
            The quantile value (y-coordinate) to annotate.
        label_offset : tuple of float
            The (x, y) offset to apply to the label position relative to the quantile point.
        ax : matplotlib.axes.Axes
            The matplotlib axes object on which to plot the annotation.
        """
        
        xpos = x[xindx]
        xoff, yoff = label_offset  
        x_mult = 1 - (xpos/self.time_max)
        print(x_mult)
        # Adjust y-offset: use a random direction to avoid overlap with other labels
        ydir = np.random.choice([-1, 1])

        ax.plot(np.concatenate([np.array([0]), xpos]), [q, q], ls=":", lw=0.5, c='0.15')
        ax.annotate(
            xy=(xpos, q),
            xytext=(xpos * xoff * x_mult, q + yoff*ydir),
            text=lbl,
            arrowprops={"arrowstyle": "-", "lw": 0.5}
        )

    def _intersect_km_curve(self, y, q):
        """
        Find the indices where a Kaplan-Meier curve crosses a specified quantile value.
        Parameters
        ----------
        y : array-like
            The y-values of the Kaplan-Meier survival curve.
        q : float
            The quantile value to find intersections with (e.g., 0.5 for the median).
        Returns
        -------
        numpy.ndarray
            Indices where the Kaplan-Meier curve crosses the quantile value.
        """
        yq = np.repeat(q, len(y))
        
        return np.argwhere(np.diff(np.sign(y - yq))).flatten()
    
    def _get_surv(self, stratify: bool=False):
        """
        Retrieve survival data, optionally stratified by an annotating feature.

        Parameters
        ----------
        stratify : bool, optional
            If True, yields survival data (time/event) for each category of the annotating feature
            specified by `self.s1`. If False, returns the complete survival data. Default is False.

        Returns
        -------
        generator or np.ndarray
            If stratify is True, returns a generator yielding survival data arrays for each category
            in `self.s1_categories`. If stratify is False, returns the full survival data array
            from `self.data.surv`.

        Notes
        -----
        - The survival data should be in the format used by scikit-survival (sksurv).
        - Assumes `self.data.surv` contains the survival data and `self.data.df[self.s1]` provides
          the grouping feature.

        """
  
        return (
            (self.data.surv[self.data.df[self.s1] == cat] for cat in self.s1_categories)
            if stratify else self.data.surv
        )
        
    def _plot_curve(self, surv, line_color, q, label_offset, ax, add_label: bool, draw_ci:bool, **step_kwargs):
        """
        Plots a Kaplan-Meier survival curve on the given axes, with optional confidence intervals and quantile labeling.
        Parameters
        ----------
        surv : pd.DataFrame
            Survival data containing event and time columns.
        line_color : str or tuple
            Color to use for the survival curve and confidence interval.
        q : float
            Quantile to label on the curve (e.g., 0.5 for median survival).
        label_offset : float
            Offset for the quantile label placement on the plot.
        ax : matplotlib.axes.Axes
            Matplotlib axes object to plot on.
        add_label : bool
            Whether to add a quantile label to the curve.
        draw_ci : bool
            Whether to draw the confidence interval around the survival curve.
        **step_kwargs
            Additional keyword arguments passed to `ax.step`.
        Returns
        -------
        None
        """
       
        x, y, conf = kme(surv[self.data.event], surv[self.data.time], conf_type="log-log")
        ax.step(x, y, where="post", color=line_color, **step_kwargs)
        if draw_ci:
            ax.fill_between(x, conf[0], conf[1], alpha=0.1, step="post", color=line_color)
        if add_label:
            self._label_quantile(x, self._intersect_km_curve(y, q), f"{np.quantile(surv[self.data.time], 1-q):.0f}", q, label_offset, ax)
    
    def _pivot_logrank(self):

        """
        Pivot the pairwise logrank test results into a DataFrame for visualization and further analysis.
        
        Returns
        -------
        tuple
            A tuple containing:
            - pd.DataFrame: Pivoted DataFrame with categories as rows and columns, containing FDR-adjusted p-values ('fdr_bh') from pairwise logrank tests.
            - np.ndarray: Flattened array of the pivoted p-values.
            - np.ndarray: Flattened array of column indices corresponding to the pivoted values.
            - np.ndarray: Flattened array of row indices corresponding to the pivoted values.
        Raises
        ------
        ValueError
            If pairwise logrank tests have not been calculated prior to calling this method.
        """
        if not hasattr(self, 'pairwise_logranks'):
            raise ValueError("Pairwise logrank tests have not been calculated. Call `calc_pairwise_logranks()` first.")
        
        pivot_df = self.pairwise_logranks.pivot_table(index='A', columns='B', values='fdr_bh')
        pivot_df = pivot_df.take(np.argsort(pivot_df.isnull().sum(axis=1)))  # order by number of NaNs
        pivot_values_flat = pivot_df.values.ravel()
        col_indices, row_indices = [arr.ravel() for arr in np.meshgrid(np.arange(pivot_df.shape[1]), np.arange(pivot_df.shape[1]))]

        return pivot_df, pivot_values_flat, col_indices, row_indices

    def kmplot(self, stratify:bool=False, draw_ci: bool=True, add_label: bool=True, q:float=0.5, label_offset: tuple=(2, 0.2), ax=None, **step_kwargs):
        """
        Plot Kaplan-Meier survival curves.

        Parameters
        ----------
        stratify : bool, optional
            If True, plot survival curves stratified by group. If False, plot overall survival curve. Default is False.
        draw_ci : bool, optional
            If True, draw confidence intervals for the survival curves. Default is True.
        add_label : bool, optional
            If True, add labels to the survival curves. Default is True.
        q : float, optional
            Quantile to use for labeling the survival curve (e.g., median survival time). Default is 0.5.
        label_offset : tuple, optional
            Offset for the label position as (x_offset, y_offset). Default is (2, 0.2).
        ax : matplotlib.axes.Axes, optional
            Matplotlib Axes object to plot on. If None, uses current axes.
        **step_kwargs
            Additional keyword arguments passed to the step plotting function.
        Returns
        -------
        None
            The function plots the survival curves on the given axes.
        """
        
        if ax is None:
            ax = plt.gca()

        if stratify:

            for strat_surv, strat_col in zip(self._get_surv(stratify=True), self.s1_colors):

                self._plot_curve(strat_surv, strat_col, q, label_offset, ax, add_label, draw_ci, **step_kwargs)

        else:

            surv_all = self._get_surv()

            self._plot_curve(surv_all, "C1", q, label_offset, ax, add_label, draw_ci, **step_kwargs)
           
        
    def add_global_p(self, cut_p: bool=True, ax=None, **legend_kwargs):
        """
        Adds a global p-value annotation (typically from a logrank test) to a matplotlib Axes as a legend.
        Parameters
        ----------
        cut_p : bool, optional (default=True)
            If True, formats the p-value using the `_cut_p` function for display (e.g., "***").
            If False, displays the p-value in scientific notation.
        ax : matplotlib.axes.Axes, optional
            The axes to which the p-value legend will be added. If None, uses the current axes (`plt.gca()`).
        **legend_kwargs : dict
            Additional keyword arguments passed to `ax.legend()` for customizing the legend appearance.
        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes object with the added p-value legend.
        Notes
        -----
        - The legend is added as an artist to the axes and displays the global logrank p-value.
        - Default legend appearance can be overridden using `legend_kwargs`.
        """
    
        if ax is None:
            ax = plt.gca()
            
        def_kwargs = {'loc':9, 'handlelength':0, 'handletextpad':0, 
        "frameon":False, 'fontsize':'small', 'labelcolor':'0.15'}
        
        if legend_kwargs:
            for k,v in legend_kwargs.items():
                def_kwargs.update({k:v})

        pval = _cut_p(self.pval) if cut_p else f"{self.pval:.1e}"

        leg = ax.legend([Patch(color='w')], [f'logrank p: {pval}'], **def_kwargs)

        ax.add_artist(leg)
        
        return ax
    
    def calc_pairwise_logranks(self):
        """
        Calculates pairwise log-rank test statistics and p-values for all combinations of categories in `self.s1_categories`.

        This method performs the following steps:
            1. Generates all unique pairs of categories from `self.s1_categories`.
            2. For each pair, filters the data to include only samples belonging to the current pair of categories.
            3. Computes the survival statistics for the filtered data.
            4. Performs a pairwise log-rank test (using `compare_survival`) between the two categories.
            5. Collects the chi-squared statistic and p-value for each pair.
            6. Adjusts the p-values for multiple testing using the Benjamini-Hochberg FDR method (`fdr`).
            7. Stores the results in a DataFrame (`self.pairwise_logranks`) with columns: 'A', 'B', 'chi2', 'pval', and 'fdr_bh'.

        Attributes:
            pairwise_logranks (pd.DataFrame): DataFrame containing the results of the pairwise log-rank tests.

        Returns:
            None: The results are stored in `self.pairwise_logranks`.
        """

        if len(self.s1_categories) <= 2:
            raise ValueError("At least 3 categories are required for pairwise logrank tests.")

        combos = list(combinations(self.s1_categories, 2))

        surv = self._get_surv()

        res = []
        for combo in combos:

            combo = list(combo)
            df_sub = self.data.df.query(f"{self.s1} in @combo")
            surv_sub = surv[df_sub.index]
            csq, pval = compare_survival(surv_sub, df_sub[self.s1])
            res.append((combo[0],combo[1], csq, pval))

        pairwise_logranks = pd.DataFrame(res, columns=['A', "B", 'chi2','pval'])

        pairwise_logranks.insert(4, "fdr_bh", fdr(pairwise_logranks['pval']))

        self.pairwise_logranks = pairwise_logranks

    def heatmap_pairwise_logranks(self, cut_p: bool=True, ax=None, mesh_kwargs: dict=None, **text_kwargs):

        """
        Plots a heatmap of pairwise log-rank test FDR values between groups.
        This method visualizes the results of pairwise log-rank tests as a heatmap, where each cell represents
        the FDR (False Discovery Rate) value for the comparison between two groups. The FDR values can be
        thresholded for display using `cut_p`. The heatmap is annotated with the FDR values, and both the
        appearance of the mesh and the text can be customized.

        Parameters
        ----------
        cut_p : bool, default=True
            If True, FDR values are thresholded for display using `_cut_p`. If False, raw FDR values are shown in scientific notation.
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot the heatmap. If None, uses the current axes.
        mesh_kwargs : dict, optional
            Additional keyword arguments passed to `ax.pcolormesh` for customizing the heatmap appearance.
        **text_kwargs
            Additional keyword arguments passed to `ax.text` for customizing the annotation text properties.
        Returns
        -------
        None

        Notes
        -----
        This method requires the results of pairwise log-rank tests to be available via `self._pivot_logrank()`.
        """
    
        if ax is None:
            ax = plt.gca()

        text_props = {"fontsize":'x-small', "ha":'center',  'va':'center'}
        mesh_props = {'cmap':'Reds_r'}

        if text_kwargs:
            for k, v in text_kwargs.items():
                text_props.update({k:v})

        if mesh_kwargs:
            for k, v in mesh_kwargs.items():
                mesh_props.update({k:v})

        heat_df, fdrs, x, y = self._pivot_logrank()
        nrow, ncol = heat_df.shape
        mesh = ax.pcolormesh(heat_df, **mesh_props)
        ax.set_xticks(np.arange(ncol)+0.5, heat_df.columns)
        ax.set_yticks(np.arange(nrow)+0.5, heat_df.index)
        ax.spines[['left', 'top']].set_visible(False)

        for xx, yy, fdr in zip(x, y, fdrs):
            if not np.isnan(fdr):
                color = _color_light_or_dark(np.array(mesh.to_rgba(fdr)))
                if cut_p:
                    fdr = _cut_p(fdr)
                else:
                    fdr = f'{fdr:.2e}'
                ax.text(xx+0.5, yy+0.5, fdr, color=color, **text_props)
        