
# As always
import numpy as np
import pandas as pd

# Lots of plotting here, so:
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.cm import coolwarm, ScalarMappable
import matplotlib.colors as mcolors

# gene set enrichment and helpers
from .aREA import aREA
from .utils import gene_sets_to_regulon, _prep_ges
from . import plotting as pl
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

"""
This module implements gene set enrichment functionality for one-tailed gene sets. 
"""

class Gsea1T:
    
    """Base class for one-tailed gene set enrichment analysis"""

    def __init__(self, 
                 ges: pd.Series, 
                 gene_set: list, 
                 weight: float = 1):

        self.weight = weight

        if not isinstance(ges, pd.Series):
            raise TypeError('Need an indexed pandas Series, please.')
        else:
            self.ges = _prep_ges(ges, asc_sort=False) # Gsea1T needs to be sorted from high to low for running sum
            self.ns = len(self.ges)
            self.along_scores = [*range(self.ns)]

        if not np.in1d(gene_set, ges.index).any():
            raise ValueError('None of the genes in gene set found in GES index')
        else:
            self.gs_org = gene_set
            self.gs_final = [g for g in gene_set if g in self.ges.index]

        self.gs_idx = self._find_hits(self.ges, self.gs_final)

        self.rs = self._derive_rs(self.ges, self.gs_idx, self.weight)
        self.es_idx = np.abs(self.rs).argmax()

        self.gs_reg = gene_sets_to_regulon({'GS':self.gs_final}, minsize=len(self.gs_final))
        self.aREA_nes = aREA(self.ges,
                            self.gs_reg, 
                            minsize=len(self.gs_final)).iloc[0][0]

        self.pval = norm.sf(np.abs(self.aREA_nes))*2
        self.ledge, self.ledge_xinfo = self._get_ledge(self.ges, self.gs_idx, self.es_idx)

    def __repr__(self):

        """String representation for the Gsea1T class"""

        return f"Gsea1T(GES length: {self.ns}, Gene set:{len(self.gs_org)}, Overlap: {len(self.gs_final)})"

    def _find_hits(self, 
                   ges: pd.Series,
                   gene_set: list)-> list:

        """Finds the positions of a gene set in a given gene expression signature list

        Parameters
        ----------
        ges :
            pd.Series: The gene expression signature (indexed numeric pandas Series).
        gene_set :
            list: The gene set in question.
        ges: pd.Series :
            
        gene_set: list :
            

        Returns
        -------
        list
            A list of positions for each gene set member in the gene expression signature.

        """

        return [idx for idx, gene in enumerate(ges.index.values) if gene in gene_set]

    def _derive_rs(self, 
                   ges: pd.Series, 
                   gene_indices: list,
                   weight: float)->np.ndarray:
        
        """Derives the running sum for plotting

        Parameters
        ----------
        ges :
            pd.Series: The gene expression signature (indexed numeric pandas Series).
        gene_indices :
            list: The positions of the gene set members in the signature.
        weight :
            float: Weighting exponent, see Subramanian PNAS 2005, for details.
        ges: pd.Series :
            
        gene_indices: list :
            
        weight: float :
            

        Returns
        -------
        np.array
            The running sum array.

        """

        Nr = np.sum(np.abs(ges[gene_indices])**weight) # normalization factor
        Nh = len(ges) - len(gene_indices) # non-hits
        tmp = np.repeat(-1/Nh, len(ges))
        tmp[gene_indices] = np.abs(ges[gene_indices])**weight/Nr
        rs = np.cumsum(tmp) # running sum
        
        return rs

    def _get_ledge(self, 
                   ges: pd.Series,
                   gene_indices: list,
                   es_index: int)-> tuple:
        """

        Parameters
        ----------
        ges :
            pd.Series: The gene expression signature (indexed numeric pandas Series).
        gene_indices :
            list:
        es_index :
            int:
        ges: pd.Series :
            
        gene_indices: list :
            
        es_index: int :
            

        Returns
        -------
        tuple
            A DataFrame of leading edge genes and their positions in the signature as well as the
        tuple
            A DataFrame of leading edge genes and their positions in the signature as well as the
            x-axis limits of the leading edge area.

        """
        # TODO: This is still a bit of a hack regarding the positions for the leading edge plot
        
        closer_end = es_index <= (len(ges)-es_index) # if True, we're closer to the left hand side
        
        if closer_end:
            # Leading edge indices
            ledge_idx = [idx for idx in gene_indices if idx <= es_index]
            ledge_xinfo = np.min(ledge_idx), es_index
        else:
            ledge_idx = [idx for idx in gene_indices if idx >= es_index]
            ledge_xinfo = es_index, np.max(ledge_idx)
            
            
        ledge_genes = ges.index[ledge_idx].values
        
        df = pd.DataFrame(zip(ledge_genes, ledge_idx), 
                          columns=['gene', 'index'])
        
        return df, ledge_xinfo

       
    def plot(self,
             figsize: tuple=(3, 3),
             conditions: tuple = ('A', 'B'),
             ges_symlog: bool = True,
             ges_stat_fmt: str = '1.0f',
             ges_type: str = None,
             ges_kw: dict = None,
             evt_kw: dict = None,
             rs_kw: dict = None
             )->mpl.figure.Figure:

        """This will return a figure object containing 3 axes displaying the gene expression signature,
        the gene set indices and the running sum line

        Parameters
        ----------
        figsize :
            tuple: Tuple of floats to specify the Figure size in inches (Default value = (2.5, 2.5):
        bar_alpha :
            float: A float between 0 and 1 to specify the alpha for the bars in the eventplot (Default value = 0.7).
        phenotypes :
            tuple:  (Default value = ('A', 'B'): A tuple of strings to specify the two phenotypes being compared.
        colors :
            tuple:  (Default value = ('.75', '#439D75'): A tuple of strings for a) the gene expression signature in the first axes and b) the
        colors :
            tuple:  (Default value = ('.75', '#439D75'): A tuple of strings for a) the gene expression signature in the first axes and b) the
            gene set and running sum in the other two axes.
        ges_type :
            str:  A string to specify the type of gene-level statistic (Default value = None)
        figsize: tuple :
             (Default value = (3)
        3) :
            
        conditions: tuple :
             (Default value = ('A')
        'B') :
            
        ges_symlog: bool :
             (Default value = True)
        ges_stat_fmt: str :
             (Default value = '1.0f')
        ges_type: str :
             (Default value = None)
        ges_kw: dict :
             (Default value = None)
        evt_kw: dict :
             (Default value = None)
        rs_kw: dict :
             (Default value = None)

        Returns
        -------
        
            A Figure Object.

        """
        
        # Some defaults
        ges_prop = {'color':'.5', 'alpha':0.25, 'linewidth':0.1}
        evt_prop = {'color': 'C0', 'alpha':0.7, 'linewidths':0.5}
        rs_prop = {'color':'C0'}

        fig = plt.figure(figsize=figsize, 
                         tight_layout=True)
        
        gs = fig.add_gridspec(3, 1, 
                              height_ratios=[2, 1, 7], 
                              hspace=0)

        # first graph
        ax1 = fig.add_subplot(gs[0])
        if ges_kw is not None:
            ges_prop.update(ges_kw)
        pl._plot_ges(self.along_scores, 
                     self.ges.values, 
                     conditions=conditions,
                     is_high_to_low=True,
                     ges_type=ges_type, 
                     symlog=ges_symlog,
                     stat_fmt=ges_stat_fmt,
                     ax=ax1, 
                     **ges_prop)
        
        # second graph: bars to indicate positions of individual genes
        ax2 = fig.add_subplot(gs[1])
        if evt_kw is not None:
            evt_prop.update(evt_kw)
        ax2.eventplot(self.gs_idx, **evt_prop)
        ax2.axis('off')

        # Third graph: Running sum
        ax3 = fig.add_subplot(gs[2])
        if rs_kw is not None:
            rs_prop.update(rs_kw)
        ax3.tick_params(labelsize='x-small')
        pl._plot_run_sum(self.rs, self.es_idx, ax=ax3, **rs_prop)
        leg = pl._stats_legend(self.aREA_nes, self.pval)
        ax3.add_artist(leg)
        pl._format_xaxis_ges(self.ns, ax=ax3)
        
        return fig

    def plot_ledge(self,
         figsize: tuple=(3.5, 3),
         highlight: tuple = None,
         rs_kw: dict = None,
         lbl_kw: dict = None,
         patch_kw:dict = None)->mpl.figure.Figure:
        """

        Parameters
        ----------
        figsize: tuple :
             (Default value = (3.5)
        3) :
            
        highlight: tuple :
             (Default value = None)
        rs_kw: dict :
             (Default value = None)
        lbl_kw: dict :
             (Default value = None)
        patch_kw:dict :
             (Default value = None)

        Returns
        -------

        """
        
        fig = plt.figure(figsize=figsize)
        
        # setup
        genes = self.ledge['gene'].values
         
        # Some defaults
        rs_prop = {'color':'C0'}
        if rs_kw is not None:
            rs_prop.update(rs_kw)
            
        lbl_prop = {'fontsize':4, 'rotation':90, 'ha':'center', 'va':'center'}
        if lbl_kw is not None:
            lbl_prop.update(lbl_kw)
                           
        # grid
        if self.aREA_nes >= 0:
            gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.1)
            ax1 = fig.add_subplot(gs[0])          
            pl._plot_ledge_labels(self.ledge_xinfo, genes=genes, ax=ax1, highlight=highlight, **lbl_prop)
            
            ax2 = fig.add_subplot(gs[1])
            pl._plot_run_sum(self.rs, self.es_idx, ax=ax2, **rs_prop)
            leg = pl._stats_legend(self.aREA_nes, self.pval)
            ax2.add_artist(leg)
            pl._format_xaxis_ges(self.ns, ax=ax2)
            pl.zoom_effect(ax1, ax2, patch_kw=patch_kw)
            
        else:
            gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.1)
            rc_new = {'xtick.bottom':False, 'xtick.labelbottom':False,
                      'xtick.top':True, 'xtick.labeltop':True}
            
            with plt.rc_context(rc_new):
                ax1 = fig.add_subplot(gs[0])
                pl._plot_run_sum(self.rs, self.es_idx, ax=ax1, **rs_prop)
                leg = pl._stats_legend(self.aREA_nes, self.pval)
                ax1.add_artist(leg)
                pl._format_xaxis_ges(self.ns, ax=ax1)
      
            ax2 = fig.add_subplot(gs[1])
            pl._plot_ledge_labels(self.ledge_xinfo, genes=genes, ax=ax2, highlight=highlight, upper=False, **lbl_prop)
            pl.zoom_effect(ax2, ax1, upper=False, patch_kw=patch_kw)
        
        return fig    

            
class Gsea1TMultSigs:
    
    """To implement one-tailed gene set enrichment analysis on multiple different signatures and visualize them"""
     
    def __init__(self, 
                 dset:pd.DataFrame, 
                 gene_set:list,  
                 ordered: bool=True):
                
        if not isinstance(dset, pd.DataFrame):
            raise TypeError('Need an indexed pandas DataFrame, please.')
            
        else: 
            self.dset = _prep_ges(dset)
            self.ns = len(self.dset)
            self.samples = self.dset.columns.values
        
        if not np.in1d(gene_set, dset.index).any():
            raise ValueError('None of the genes in gene set found in dset DataFrame index')
        else:
            self.gs_org = gene_set
            self.gs_final = [g for g in gene_set if g in self.dset.index]
        
        self.gs_idx = self._find_hits(self.dset, self.gs_final)
        
        self.gs_reg = gene_sets_to_regulon({'GS':self.gs_org}, minsize=len(self.gs_final))
        
        self.stats = self._get_stats(dset=self.dset, 
                                     regulon=self.gs_reg, 
                                     minsize=len(self.gs_final),
                                     samples=self.samples)
        
        if ordered:
            idx = self.stats['NES'].argsort().values
            self.stats = self.stats.take(idx, axis=0)
            self.stats.reset_index(inplace=True, drop=True)
            self.gs_idx = self.gs_idx.take(idx, axis=0)
            
    
    def __repr__(self):
        
        """String representation for the Gsea1TMultSig class"""

        return (
                f"Gsea1TMult(Number of genes: {self.ns}\n"
                f"Number of signatures: {len(self.samples)}\n"
                f"Gene set: n={len(self.gs_org)}\n"
                f"Overlap: n={len(self.gs_final)}\n"
                )    
       
    def _find_hits(self, dset, *gene_sets):
        
        """Finds the positions of a gene set in a given gene expression signaure matrix

        Parameters
        ----------
        dset :
            
        *gene_sets :
            

        Returns
        -------

        """
        
        # rank order all signatures 
        hits = dset.apply(lambda x: x.sort_values().argsort(), axis=0)
        
        # subset to those in the gene set
        hitlist = [hits.loc[gene_set].values.T for gene_set in gene_sets]
        
        col_names = [f'position_set_{i}' for i in range(len(gene_sets))]
        
        hitdf = pd.DataFrame(zip(*hitlist), index=self.samples, columns=col_names)
        # Return the indices for the gene set in each signature as a 
        # numpy array of arrays, transverse for plotting
        return hitdf
    
    def _get_stats(self, 
                   dset: pd.DataFrame, 
                   regulon: pd.DataFrame, 
                   minsize: int,
                   samples:np.ndarray):
        
        """Computes normalized enrichment scores and some tools for later visualization

        Parameters
        ----------
        dset: pd.DataFrame :
            
        regulon: pd.DataFrame :
            
        minsize: int :
            
        samples:np.ndarray :
            

        Returns
        -------

        """
        
        nes = aREA(dset, regulon, minsize).iloc[0,:].values # get a flattened one-dimensional array
        pvals = norm.sf(np.abs(nes))*2 # retrieve two-sided p-value from normal distribution
        fdrs = multipletests(pvals, method = 'fdr_bh')[1] #FDR
        
        stats = pd.DataFrame(data=zip(samples, nes, pvals, fdrs),
                            columns=['signature_name', 'NES', 'pval', 'FDR'])

        return stats
        
        
    def plot(self, 
             figsize: tuple = (3, 3),
             add_title: str = None,
             evt_kw: dict = None,
             norm_kw: dict = None,
             pcm_kw: dict= None):
         
        """This will return a figure object containing 4 axes displaying the gene expression signature,
        the gene set indices and the running sum line

        Parameters
        ----------
        bar_alpha :
            float:  (Default value = 0.5)
        figsize :
            tuple:  (Default value = (3.5)
        3) :
            
        norm_kws :
            dict:  (Default value = None)
        figsize: tuple :
             (Default value = (3)
        add_title: str :
             (Default value = None)
        evt_kw: dict :
             (Default value = None)
        norm_kw: dict :
             (Default value = None)
        pcm_kw: dict :
             (Default value = None)

        Returns
        -------

        """
        
        df = self.stats.copy()
        
        # needs to be unnested
        evt_data = [arr for tup in self.gs_idx.values for arr in tup]
        
        evt_prop = {'linelengths':3/4, 'color':'.15', 'alpha':0.7, 'linewidths':0.5}
             
        norm, pcm_prop = pl._prepare_nes_colors(df, norm_kw, pcm_kw)
        
        # Prettify FDR format
        df['FDR'] = df['FDR'].apply(pl._fdr_formatter)
                
        # In this first bit, we determine the NES color scheme
        
        fig = plt.figure(figsize=figsize) 
        
        # height ratio may change according to figsize and number of signatures,
        # carve out a fixed number of inches. 0.15 translates to roughly 0.38 cm. 
        height_clb = 0.15
        rest = figsize[1]-height_clb
                
        gs = fig.add_gridspec(nrows=2, 
                      ncols=2, 
                      hspace=0.025, 
                      wspace=0.025,
                      width_ratios=[10,1],
                      height_ratios=[height_clb, rest])
        
        # Plot 1: Illustrate the ordered signature as a colorbar
        ax1 = fig.add_subplot(gs[0,0])
        cb = fig.colorbar(ScalarMappable(cmap=plt.cm.RdBu_r), 
             orientation='horizontal', ticks=[], cax=ax1)
        cb.outline.set_visible(False)
        ax1.text(0, 0.5, 'Low', color='w', ha='left', va='center', fontsize='x-small')
        ax1.text(1, 0.5, 'High', color='w', ha='right', va='center', fontsize='x-small')
        if add_title is not None:
            ax1.set_title(add_title, fontsize='x-small')
        
        # Plot 2: Illustrate targets
        ax2 = fig.add_subplot(gs[1,0])
        
        if evt_kw is not None:
            evt_prop.update(evt_kw)
            
        ax2.eventplot(evt_data, **evt_prop)
        ax2.set_yticks(range(len(df)))
        ax2.set_ylim([-0.5, len(df)-0.5])
        ax2.set_yticklabels(df['signature_name'].values, fontsize='x-small')
        ax2.set_xlim([-.5, self.ns+.5])
        # x-axis formating
        pl._format_xaxis_ges(self.ns, ax=ax2)    
        
        # remove spines
        for spine in ['right', 'left', 'top']:
            ax2.spines[spine].set_visible(False)

        # Plot 3: NES heatmap
        ax3 = fig.add_subplot(gs[1,1])
    
        ax3.pcolormesh(df['NES'].values[:,np.newaxis], 
               norm=norm, **pcm_prop)

        for row in df.itertuples():
            ax3.text(0.5, row.Index + 0.5, f'{row.NES:1.1f}',
                fontsize='xx-small', c=row.color,
                ha='center', va='center')
            ax3.text(1, row.Index + 0.5, row.FDR,
                fontsize='xx-small', c='k',
                ha='left', va='center')
            ax3.axis('off')

        # Plot 4: Not sure if necessary to open new plot, but use grid fully. Just titles for annotation
        ax4 = fig.add_subplot(gs[0,1])
        ax4.text(0.5, 0, 'NES', fontsize='x-small', c='k', ha='center', va='bottom')
        ax4.text(1.1, 0, 'FDR', fontsize='x-small', c='k', ha='left', va='bottom')
        ax4.axis('off')
        
        return fig



class Gsea1TMultSets:
    
    """To implement one-tailed gene set enrichment of multiple gene sets on one gene expression signature
    and visualize the results

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, 
                 ges: pd.Series, 
                 gene_sets: dict, 
                 ordered: bool = True,
                 minsize: int = 20):

        self.minsize = minsize

        if not isinstance(ges, pd.Series):
            raise TypeError('Need an indexed pandas Series, please.')
        if not isinstance(gene_sets, dict):
            raise TypeError('Need a dictionary of gene sets, please')
        else:
            self.ges = _prep_ges(ges)
            self.ns = len(self.ges)
            self.along_scores = [*range(self.ns)]

        self.n_org_gene_sets = len(gene_sets)
        self.gene_sets = self._prep_gene_sets(gene_sets, self.ges.index, self.minsize)
        
        if len(self.gene_sets) == 0:
            raise ValueError('None of the gene sets have enough genes to be found in GES index')
       
        self.gs_names = list(self.gene_sets.keys())
        
        self.gs_reg = gene_sets_to_regulon(self.gene_sets, minsize=self.minsize)
              
        self.gs_idx = self._find_hits(self.ges, self.gene_sets)
        
        self.stats = self._get_stats(self.ges, self.gs_reg, self.minsize, self.gs_idx)
                
        if ordered:
            self.stats.sort_values('NES', inplace=True)
            self.stats.reset_index(inplace=True, drop=True)
            

    def __repr__(self):

        """String representation for the Gsea1TMultSet class"""

        return (
                f"Gsea1TMultSet(GES length: {self.ns}\n"
                f"Gene sets in: {self.n_org_gene_sets}\n"
                f"Gene sets used: {len(self.gene_sets)}"
                )
        
    def _prep_gene_sets(self, 
                        gene_sets: dict,
                        genes: pd.Index,
                        minsize: int) -> dict:
        
        """Go through the provided dict of gene sets and check whether at least minsize
        genes of said gene set occur in the gene expression signature

        Parameters
        ----------
        gene_sets :
            dict:
        genes :
            pd.Index:
        minsize :
            int:
        gene_sets: dict :
            
        genes: pd.Index :
            
        minsize: int :
            

        Returns
        -------

        """
        return {key:val for key, val in gene_sets.items() if np.in1d(val, genes).sum() >= minsize}

    def _find_hits(self, 
                   ges: pd.Series, 
                   gene_sets: dict):
        """

        Parameters
        ----------
        ges :
            pd.Series:
        gene_sets :
            dict:
        ges: pd.Series :
            
        gene_sets: dict :
            

        Returns
        -------

        """

        # This will have to be sorted first, which is the case in this Class
        ges_idx = ges.copy().argsort()
        
        ges_pos = [ges_idx[ges_idx.index.intersection(val)].values for val in gene_sets.values()]
        
        return pd.DataFrame(zip(gene_sets.keys(), ges_pos), columns=['gene_set', 'positions'])
            
        
    def _get_stats(self, 
                   ges: pd.Series,
                   regulon: pd.DataFrame,
                   minsize: int,
                   add_positions: pd.DataFrame=None):
        
        """Computes normalized enrichment scores and some tools for later visualization

        Parameters
        ----------
        ges :
            pd.Series:
        regulon :
            dict:
        ges: pd.Series :
            
        regulon: pd.DataFrame :
            
        minsize: int :
            
        add_positions: pd.DataFrame :
             (Default value = None)

        Returns
        -------

        """
        
        nes = aREA(ges, regulon, minsize).iloc[:,0]
        pvals = norm.sf(np.abs(nes.values))*2 # retrieve two-sided p-value from normal distribution
        fdrs = multipletests(pvals, method = 'fdr_bh')[1] #FDR

        df = pd.DataFrame(data=zip(nes.index.values, nes.values, pvals, fdrs),
                            columns=['gene_set', 'NES', 'pval', 'FDR']) 
        
        if add_positions is not None:
            df = df.merge(add_positions, on='gene_set')
            
        return df
                    
                    
    def _filter_multset_stats(self, 
                              stats:pd.DataFrame, 
                              subset: dict): 
        
        """Filter stat results if needed for plotting

        Parameters
        ----------
        stats:pd.DataFrame :
            
        subset: dict :
            

        Returns
        -------

        """
        if not isinstance(subset, dict):
            raise TypeError('Subset instructions need to be supplied as dictionary')
        
        if not any([k in subset for k in ['FDR', 'gene_sets', 'NES']]):
            raise ValueError('Can only subset based on FDR, NES or gene set names')
            
        if 'gene_sets' in subset:
            stats = stats[stats['gene_set'].isin(subset.get('gene_sets'))]
  
        elif 'NES' in subset:
            stats = stats[stats['NES'].abs()>=subset.get('NES', 2)]
        
        else: # In the end, we will always filter based on FDR (even if you don't spell right)
            stats = stats[stats['FDR']<=subset.get('FDR', 0.1)]
        
        if not len(stats)>0:
            raise AssertionError('None of the gene sets fullfilled the required cutoff')
        
        stats.reset_index(inplace=True, drop=True)
        
        return stats

    def plot(self, 
             figsize: tuple = (3, 3),
             conditions: tuple = ('A', 'B'),
             strip_gs_names: bool = True,
             ges_symlog: bool= True,
             ges_stat_fmt: str = '1.0f',
             subset: dict = None,
             ges_type:str = None,
             ges_kw: dict = None,
             evt_kw: dict = None,
             pcm_kw: dict = None,
             norm_kw: dict = None
             ):
         
        """This will return a figure object containing 4 axes displaying the gene expression signature,
        the gene set indices and the Normalized Enrichment scores and associated FDR

        Parameters
        ----------
        phenotypes :
            tuple:  (Default value = ('A')
        'B') :
            
        bar_alpha :
            float:  (Default value = 0.5)
        figsize :
            tuple:  (Default value = (3.5)
        3) :
            
        norm_kws :
            dict:  (Default value = None)
        ges_type :
            str:  (Default value = None)
        figsize: tuple :
             (Default value = (3)
        conditions: tuple :
             (Default value = ('A')
        strip_gs_names: bool :
             (Default value = True)
        ges_symlog: bool :
             (Default value = True)
        ges_stat_fmt: str :
             (Default value = '1.0f')
        subset: dict :
             (Default value = None)
        ges_type:str :
             (Default value = None)
        ges_kw: dict :
             (Default value = None)
        evt_kw: dict :
             (Default value = None)
        pcm_kw: dict :
             (Default value = None)
        norm_kw: dict :
             (Default value = None)

        Returns
        -------

        """
        
          # Key input data: 
        df = self.stats.copy() 
        
        # Some setup: 
        ges_prop = {'color':'.5', 'alpha':0.25, 'linewidth':0.1}
        evt_prop = {'color': '.15', 'alpha':0.7, 'linelengths':3/4, 'linewidths':0.5}         
        
        norm, pcm_prop = pl._prepare_nes_colors(df, norm_kw, pcm_kw)        
                    
        if subset is not None:
            df = self._filter_multset_stats(df, subset)
            
        if strip_gs_names:
            df['gene_set'] = df['gene_set'].str.replace('^[A-Z]+_', '', ).values         
                
        df['FDR'] = df['FDR'].apply(pl._fdr_formatter)
        
        # Setup figure and gridspec
        fig = plt.figure(figsize=figsize) 
        
         # height ratio may change according to figsize and number of signatures,
        # carve out a fixed number of inches. Here: roughly 1.5 cm. 
        height_ges = 0.6
        rest = figsize[1]-height_ges
        
        gs = fig.add_gridspec(nrows=2, 
                      ncols=2, 
                      hspace=0.025, 
                      wspace=0.025,
                      width_ratios=[10,1],
                      height_ratios=[height_ges, rest])
        
        # Plot 1: Illustrate the ordered signature
        ax1 = fig.add_subplot(gs[0,0])
        if ges_kw is not None:
            ges_prop.update(ges_kw)
        pl._plot_ges(self.along_scores, self.ges.values, ges_type=ges_type, conditions=conditions, 
                     is_high_to_low=False, symlog=ges_symlog, stat_fmt=ges_stat_fmt, ax=ax1, **ges_prop)
        
        # Plot 2: Illustrate targets
        ax2 = fig.add_subplot(gs[1,0])
        if evt_kw is not None:
            evt_prop.update(evt_kw)
        ax2.eventplot(df['positions'].values, **evt_prop)
        # y-axis formatting
        ax2.set_yticks(range(len(df)))
        ax2.set_ylim([-0.5, len(df)-0.5])
        ax2.set_yticklabels(df['gene_set'], fontsize='xx-small')
        # x-axis formatting        
        ax2.set_xlim([-.5, self.ns+.5])
        pl._format_xaxis_ges(self.ns, ax=ax2)
        
        # remove spines
        for spine in ['right', 'left', 'top']:
            ax2.spines[spine].set_visible(False)

        # Plot 3: NES heatmap            
        ax3 = fig.add_subplot(gs[1,1])

        ax3.pcolormesh(df['NES'].values[:,np.newaxis], 
                    norm=norm, **pcm_prop)
    
        for row in df.itertuples():
            ax3.text(0.5, row.Index + 0.5, f"{row.NES:1.1f}",
                fontsize='xx-small', c=row.color,
                ha='center', va='center')
            ax3.text(1, row.Index + 0.5, row.FDR,
                fontsize='xx-small', c='k',
                ha='left', va='center')
            ax3.axis('off')

        # Plot 4: Not sure if necessary to open new plot, but use grid fully. Just titles for annotation
        ax4 = fig.add_subplot(gs[0,1])
        ax4.text(0.5, 0, 'NES', fontsize='x-small', c='k', ha='center', va='bottom')
        ax4.text(1.1, 0, 'FDR', fontsize='x-small', c='k', ha='left', va='bottom')
        ax4.axis('off')
        
        return fig

