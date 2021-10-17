
# As always
from typing import Sequence, Union
import numpy as np
import pandas as pd

# Lots of plotting here, so:
from matplotlib import pyplot as plt
from matplotlib.cm import coolwarm, ScalarMappable
import matplotlib.colors as mcolors
from matplotlib import ticker
from pandas.core.frame import DataFrame

# gene set enrichment and helpers
from .aREA import aREA
from ._aREA_utils import gene_sets_to_regulon, _prep_ges
from . import plotting as pl
from .Gsea1T import Gsea1T, Gsea1TMultSigs, Gsea1TMultSets
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

"""
This module implements gene set enrichment functionality for two-tailed gene sets. 
"""

class Gsea2T(Gsea1T):
    """ """

    def __init__(self, 
                 ges: pd.Series, 
                 gene_set_1: list, 
                 gene_set_2: list,
                 weight: float = 1):
               
        self.weight = weight

        if not isinstance(ges, pd.Series):
            raise TypeError('Need an indexed pandas Series, please.')
        else:
            self.ges = _prep_ges(ges, asc_sort=False) # Gsea1T needs to be sorted from high to low for running sum
            self.ns = len(self.ges)
            self.along_scores = [*range(self.ns)]

        if not np.in1d(gene_set_1, ges.index).any():
            raise ValueError('None of the genes in gene set 1 found in GES index')
        elif not np.in1d(gene_set_2, ges.index).any():
            raise ValueError('None of the genes in gene set 2 found in GES index')
        else:
            self.gs_org_1 = gene_set_1
            self.gs_final_1 = [g for g in gene_set_1 if g in self.ges.index]
            self.gs_org_2 = gene_set_2
            self.gs_final_2 = [g for g in gene_set_2 if g in self.ges.index]
        
        self.gs_idx_1 = self._find_hits(self.ges, self.gs_final_1)
        self.gs_idx_2 = self._find_hits(self.ges, self.gs_final_2)

        self.rs_1 = self._derive_rs(self.ges, self.gs_idx_1, self.weight)
        self.es_idx_1 = np.abs(self.rs_1).argmax()
        
        self.rs_2 = self._derive_rs(self.ges, self.gs_idx_2, self.weight)
        self.es_idx_2 = np.abs(self.rs_2).argmax()

        self.gs_reg_1 = gene_sets_to_regulon({'GS1':self.gs_final_1}, minsize=len(self.gs_final_1))
        self.aREA_nes_1 = aREA(self.ges,
                            self.gs_reg_1).iloc[0][0]
        
        self.gs_reg_2 = gene_sets_to_regulon({'GS2':self.gs_final_2}, minsize=len(self.gs_final_2))
        self.aREA_nes_2 = aREA(self.ges,
                            self.gs_reg_2).iloc[0][0]

        self.pval_1 = norm.sf(np.abs(self.aREA_nes_1))*2
        self.ledge_1, self.ledge_xinfo_1 = self._get_ledge(self.ges, 
                                                           self.gs_idx_1, 
                                                           self.es_idx_1)
        
        self.pval_2 = norm.sf(np.abs(self.aREA_nes_2))*2
        self.ledge_2, self.ledge_xinfo_2 = self._get_ledge(self.ges, 
                                                           self.gs_idx_2, 
                                                           self.es_idx_2)
        
    def __repr__(self):
        
        """String representation for the Gsea2T class"""

        return (
            f"Gsea2T(GES length: {self.ns}\n"
            f"Gene set 1: {len(self.gs_org_1)}, Overlap 1: {len(self.gs_final_1)}\n"
            f"Gene set 2: {len(self.gs_org_2)}, Overlap 2: {len(self.gs_final_2)})\n"
        )
    
    def plot(self,
             figsize: tuple=(3, 3),
             conditions: tuple = ('A', 'B'),
             ges_type: str = None,
             ges_symlog: bool=True,
             ges_stat_fmt:str='1.0f',
             ges_kw: dict = None,
             evt_kw_1: dict = None,
             rs_kw_1: dict = None,
             leg_kw_1:dict=None,
             evt_kw_2: dict = None,
             rs_kw_2: dict = None,
             leg_kw_2:dict=None
             ):
        """

        Parameters
        ----------
        figsize: tuple :
             (Default value = (3)
        3) :
            
        conditions: tuple :
             (Default value = ('A')
        'B') :
            
        ges_type: str :
             (Default value = None)
        ges_symlog: bool :
             (Default value = True)
        ges_stat_fmt:str :
             (Default value = '1.0f')
        ges_kw: dict :
             (Default value = None)
        evt_kw_1: dict :
             (Default value = None)
        rs_kw_1: dict :
             (Default value = None)
        leg_kw_1:dict :
             (Default value = None)
        evt_kw_2: dict :
             (Default value = None)
        rs_kw_2: dict :
             (Default value = None)
        leg_kw_2:dict :
             (Default value = None)

        Returns
        -------

        """
        
        # Some defaults
        ges_prop = {'color':'.5', 'alpha':0.25, 'linewidth':0.1}
        evt_prop_1 = {'color': '#AC3220', 'alpha':0.7, 'linewidths':0.5} # Chinese Red
        evt_prop_2 = {'color': '#50808E', 'alpha':0.7, 'linewidths':0.5} # Teal Blue
        rs_prop_1 = {k:v for k, v in evt_prop_1.items() if k=='color'}
        rs_prop_2 = {k:v for k, v in evt_prop_2.items() if k=='color'}
        leg_prop_1 = {'title':'Set1', "loc":1, 'labelcolor':evt_prop_1.get('color')}  
        leg_prop_2 = {'title':'Set2', "loc":3, 'labelcolor':evt_prop_2.get('color')}
  
        fig = plt.figure(figsize=figsize, 
                         tight_layout=True)
        
        gs = fig.add_gridspec(4, 1, 
                              height_ratios=[2, 1, 6, 1], 
                              hspace=0)

        # first graph
        ax1 = fig.add_subplot(gs[0])
        if ges_kw is not None:
            ges_prop.update(ges_kw)
        pl._plot_ges(self.along_scores, 
                     self.ges.values, 
                     ges_type=ges_type,
                     conditions=conditions,
                     is_high_to_low=True,
                     symlog=ges_symlog,
                     stat_fmt=ges_stat_fmt,
                     ax=ax1, 
                     **ges_prop)
        
        # second graph: bars to indicate positions of FIRST GENE SET genes
        ax2 = fig.add_subplot(gs[1])
        if evt_kw_1 is not None:
            evt_prop_1.update(evt_kw_1)
        ax2.eventplot(self.gs_idx_1, **evt_prop_1)
        ax2.axis('off')

        # Third graph: Running sums
        ax3 = fig.add_subplot(gs[2])
        if leg_kw_1 is not None:
            leg_prop_1.update(leg_kw_1)
        if leg_kw_2 is not None:
            leg_prop_2.update(leg_kw_2)
    
        if rs_kw_1 is not None:
            rs_prop_1.update(rs_kw_1)
        if rs_kw_2 is not None:
            rs_prop_2.update(rs_kw_2)
            
        # ax3.tick_params(labelsize='x-small')
        pl._plot_run_sum(self.rs_1, self.es_idx_1, **rs_prop_1)
        leg1 = pl._stats_legend(self.aREA_nes_1, self.pval_1, leg_kw=leg_prop_1)
        ax3.add_artist(leg1)
        pl._plot_run_sum(self.rs_2, self.es_idx_2, ax=ax3, add=True, **rs_prop_2)
        leg2 = pl._stats_legend(self.aREA_nes_2, self.pval_2, leg_kw=leg_prop_2)
        ax3.add_artist(leg2)
        ax3.spines['bottom'].set_visible(False)
        
        # fourth graph: bars to indicate positions of SECOND GENE SET genes
        ax4 = fig.add_subplot(gs[3])
        if evt_kw_2 is not None:
            evt_prop_2.update(evt_kw_2)
        ax4.eventplot(self.gs_idx_2, **evt_prop_2)
        ax4.set_yticks([])
        ax4.tick_params(labelsize='x-small')
        pl._format_xaxis_ges(self.ns, ax=ax4)
      
        for spine in ['left', 'right']:
            ax4.spines[spine].set_visible(False)
        
        return fig
    
    def plot_ledge(self,
         figsize: tuple=(3, 3),
         lbl_kw_1: dict = None,
         ledge_trim_1:int=None,
         rs_kw_1: dict = None,
         leg_kw_1:dict=None,
         patch_kw_1:dict=None,
         highlight_set_1: tuple = None,
         lbl_kw_2:dict =None,
         ledge_trim_2:int=None,
         rs_kw_2:dict = None,
         leg_kw_2:dict=None,
         patch_kw_2:dict=None,
         highlight_set_2:tuple=None,):
        """

        Parameters
        ----------
        figsize: tuple :
             (Default value = (3)
        3) :
            
        lbl_kw_1: dict :
             (Default value = None)
        ledge_trim_1:int :
             (Default value = None)
        rs_kw_1: dict :
             (Default value = None)
        leg_kw_1:dict :
             (Default value = None)
        patch_kw_1:dict :
             (Default value = None)
        highlight_set_1: tuple :
             (Default value = None)
        lbl_kw_2:dict :
             (Default value = None)
        ledge_trim_2:int :
             (Default value = None)
        rs_kw_2:dict :
             (Default value = None)
        leg_kw_2:dict :
             (Default value = None)
        patch_kw_2:dict :
             (Default value = None)
        highlight_set_2:tuple :
             (Default value = None)

        Returns
        -------

        """
        
        fig = plt.figure(figsize=figsize)
        
        # setup
        genes_1 = self.ledge_1['gene'].values
        genes_2 = self.ledge_2['gene'].values
              
        # DEFAULTS, Patch will always follow running sum color
        rs_prop_1 = {'color':'#AC3220'} ; patch_prop_1 =  rs_prop_1.copy() # Chinese red
        if rs_kw_1 is not None:
            rs_prop_1.update(rs_kw_1)
        if patch_kw_1 is not None:
            patch_prop_1.update(patch_kw_1)
            
        rs_prop_2 = {'color':'#50808E'} ; patch_prop_2 = rs_prop_2.copy() # Teal Blue 
        if rs_kw_2 is not None:
            rs_prop_2.update(rs_kw_2)
        if patch_kw_2 is not None:
            patch_prop_2.update(patch_kw_2)
    
        lbl_prop_1 = {'fontsize':4, 'rotation':90, 'ha':'center', 'va':'center'}
        lbl_prop_2 = lbl_prop_1.copy()
        if lbl_kw_1 is not None:
            lbl_prop_1.update(lbl_kw_1)
        if lbl_kw_2 is not None:
            lbl_prop_2.update(lbl_kw_2)
        
        # LEGEND DEFAULT
        leg_prop_1 = {'title':'Set1', "loc":1, 'labelcolor':rs_prop_1.get('color')} 
        if leg_kw_1 is not None:
            leg_prop_1.update(leg_kw_1)
        leg_prop_2 = {'title':'Set2', "loc":3, 'labelcolor':rs_prop_2.get('color')}
        if leg_kw_2 is not None:
            leg_prop_2.update(leg_kw_2)
      
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 4, 1], hspace=0.1)
        
        # First gene set leading edge, above
        ax1 = fig.add_subplot(gs[0])          
        pl._plot_ledge_labels(self.ledge_xinfo_1, genes=genes_1, ax=ax1, 
                              highlight=highlight_set_1, trim_ledge=ledge_trim_1, **lbl_prop_1)
        
        # Running sums
        ax2 = fig.add_subplot(gs[1])
        pl._plot_run_sum(self.rs_1, self.es_idx_1, ax=ax2, **rs_prop_1)
        leg1 = pl._stats_legend(self.aREA_nes_1, self.pval_1, leg_kw=leg_prop_1)
        ax2.add_artist(leg1)
        pl._plot_run_sum(self.rs_2, self.es_idx_2, ax=ax2, add=True, **rs_prop_2)
        leg2 = pl._stats_legend(self.aREA_nes_2, self.pval_2, leg_kw=leg_prop_2)
        ax2.add_artist(leg2)
        ax2.set_xticks([])
        
        ax3 = fig.add_subplot(gs[2])   
        pl._plot_ledge_labels(self.ledge_xinfo_2, genes=genes_2, ax=ax3, upper=False, 
                              highlight=highlight_set_2, trim_ledge=ledge_trim_2, **lbl_prop_2)
        
        # Zooming
        pl.zoom_effect(ax1, ax2, patch_kw=patch_prop_1)
        pl.zoom_effect(ax3, ax2, upper=False, patch_kw=patch_prop_2)    
        
        return fig    


class GseaReg(Gsea1T):
    """ """

    def __init__(self, 
                 ges: pd.Series, 
                 regulon: pd.DataFrame):

        if not isinstance(ges, pd.Series):
            raise TypeError('Need an indexed pandas Series, please.')
        else:
            self.ges = _prep_ges(ges, asc_sort=False) # Gsea1T needs to be sorted from high to low for running sum
            self.ns = len(self.ges)
            self.along_scores = [*range(self.ns)]
            
            
        if not isinstance(regulon, pd.DataFrame):
            raise TypeError('Regulon needs to be supplied as a DataFrame, please.')
        else:
            self.regulator = ''.join(set(regulon['source']))
            self.regulon_in = regulon
            
        # if not all(self.regulon_in.columns.values == ['source',' target', 'mor', 'likelihood']):
        #     raise ValueError('Regulon value DataFrame must have columns: source, target, mor, likelihood')
    
        if not np.in1d(self.regulon_in['target'].values, self.ges.index).any():
            raise ValueError("None of the regulator targets were found in the gene expression signature index!")
        else:
            self.reg_df, self.reg_pos, self.reg_neg = self._split_regulon(self.regulon_in, self.ges.index)

        # Gene set 1 are the positive targets
        self.gs_idx_1 = self._find_hits(self.ges, self.reg_pos['target'].to_list())
        self.gs_idx_2 = self._find_hits(self.ges, self.reg_neg['target'].to_list())

        self.rs_1 = self._derive_rs(self.ges, self.gs_idx_1, 1) #Default weight, cannot be modified
        self.es_idx_1 = np.abs(self.rs_1).argmax()
        
        self.rs_2 = self._derive_rs(self.ges, self.gs_idx_2, 1)
        self.es_idx_2 = np.abs(self.rs_2).argmax()

        self.aREA_nes = aREA(self.ges, self.regulon_in, minsize=len(self.reg_df)).iloc[0][0]
        self.pval = norm.sf(np.abs(self.aREA_nes))*2
        
        self.ledge_1, self.ledge_xinfo_1 = self._get_ledge(self.ges, 
                                                           self.gs_idx_1, 
                                                           self.es_idx_1)
        
        self.ledge_2, self.ledge_xinfo_2 = self._get_ledge(self.ges, 
                                                           self.gs_idx_2, 
                                                           self.es_idx_2)
        
    def __repr__(self):
        
        """String representation for the GseaReg class"""

        return (
            f"GseaReg(GES length: {self.ns}\n"
            f"Regulator: {self.regulator}\n"
            f"Number of targets: {len(self.regulon_in)}, Overlap with GES: {len(self.reg_df)}\n"
            f"Positive targets: {len(self.reg_pos)}, Negative targets: {len(self.reg_neg)}\n"
        )
    
    def _split_regulon(self, 
                       reg_df:pd.DataFrame, 
                       ges_index: pd.Index):
        """This will divide the regulon into negative and positive targets

        Parameters
        ----------
        reg_df:pd.DataFrame :
            
        ges_index: pd.Index :
            

        Returns
        -------

        """
        # prefiltering
        reg_df_new = reg_df[reg_df['target'].isin(ges_index)] # We test that there must be some before, so safe to subset. 
        
        # In theory, it's not necessary to have both negative and positive targets. 
        # But this is about plotting, so we require both. 
        mask = reg_df_new['mor'].values>=0
        if all(mask) or not any(mask):
            raise ValueError("Could not identify two tails in the gene expression signature, please double check.")
            
        return reg_df_new, reg_df_new[mask], reg_df_new[~mask]
        
    
    def plot(self,
             figsize: tuple=(3, 3),
             conditions: tuple = ('A', 'B'),
             ges_type: str = None,
             ges_symlog: bool=True,
             ges_stat_fmt:str='1.0f',
             ges_kw: dict = None,
             evt_kw_1: dict = None,
             rs_kw_1: dict = None,
             evt_kw_2: dict = None,
             rs_kw_2: dict = None,
             stat_kw:dict =None,
             leg_kw:dict=None
             ):
        """

        Parameters
        ----------
        figsize: tuple :
             (Default value = (3)
        3) :
            
        conditions: tuple :
             (Default value = ('A')
        'B') :
            
        ges_type: str :
             (Default value = None)
        ges_symlog: bool :
             (Default value = True)
        ges_stat_fmt:str :
             (Default value = '1.0f')
        ges_kw: dict :
             (Default value = None)
        evt_kw_1: dict :
             (Default value = None)
        rs_kw_1: dict :
             (Default value = None)
        evt_kw_2: dict :
             (Default value = None)
        rs_kw_2: dict :
             (Default value = None)
        stat_kw:dict :
             (Default value = None)
        leg_kw:dict :
             (Default value = None)

        Returns
        -------

        """
        
        # Some defaults
        ges_prop = {'color':'.5', 'alpha':0.25, 'linewidth':0.1}
        evt_prop_1 = {'color': '#AC3220', 'alpha':0.7, 'linewidths':0.5} # Chinese Red
        if evt_kw_1 is not None:
            evt_prop_1.update(evt_kw_1)        
        evt_prop_2 = {'color': '#50808E', 'alpha':0.7, 'linewidths':0.5} # Teal Blue
        if evt_kw_2 is not None:
            evt_prop_2.update(evt_kw_2)
            
        rs_prop_1 = {'color':evt_prop_1.get('color')}
        rs_prop_2 = {'color':evt_prop_2.get('color')}
        
        stat_prop = {'loc':3, "title":self.regulator} # Stats go to bottom left by default
        if stat_kw is not None:
            stat_prop.update(stat_kw)

        fig = plt.figure(figsize=figsize, 
                         tight_layout=True)
        
        gs = fig.add_gridspec(4, 1, 
                              height_ratios=[2, 1, 6, 1], 
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
        
        # second graph: bars to indicate positions of FIRST GENE SET genes
        ax2 = fig.add_subplot(gs[1])
        ax2.eventplot(self.gs_idx_1, **evt_prop_1)
        ax2.axis('off')

        # Third graph: Running sums
        
        ax3 = fig.add_subplot(gs[2])
        if rs_kw_1 is not None:
            rs_prop_1.update(rs_kw_1)
        if rs_kw_2 is not None:
            rs_prop_2.update(rs_kw_2)
            
        pl._plot_run_sum(self.rs_1, self.es_idx_1, ax=ax3, **rs_prop_1)
        pl._plot_run_sum(self.rs_2, self.es_idx_2, ax=ax3, add=True, **rs_prop_2)
        leg = pl._add_reg_legend(evt_prop_1.get('color'), evt_prop_2.get('color'), leg_kw=leg_kw)
        ax3.add_artist(leg)    
        stats = pl._stats_legend(self.aREA_nes, self.pval, leg_kw=stat_prop)
        ax3.add_artist(stats)        
        ax3.set_xticks([])
        ax3.spines['bottom'].set_visible(False)
        
        # fourth graph: bars to indicate positions of SECOND GENE SET genes
        ax4 = fig.add_subplot(gs[3])
     
        ax4.eventplot(self.gs_idx_2, **evt_prop_2)
        ax4.set_yticks([])
        ax4.tick_params(labelsize='x-small')
        pl._format_xaxis_ges(self.ns, ax=ax4)
        for spine in ['left', 'right']:
            ax4.spines[spine].set_visible(False)
        
        return fig
    
    def plot_ledge(self,
         figsize: tuple=(3, 3),
         stat_kw:dict=None,
         leg_kw:dict=None,
         lbl_kw_1: dict = None,
         ledge_trim_1:int=None,
         rs_kw_1: dict = None,
         patch_kw_1:dict=None,
         highlight_set_1: tuple = None,
         lbl_kw_2:dict =None,
         ledge_trim_2:int=None,
         rs_kw_2:dict = None,
         patch_kw_2:dict=None,
         highlight_set_2:tuple=None,):
        """

        Parameters
        ----------
        figsize: tuple :
             (Default value = (3)
        3) :
            
        stat_kw:dict :
             (Default value = None)
        leg_kw:dict :
             (Default value = None)
        lbl_kw_1: dict :
             (Default value = None)
        ledge_trim_1:int :
             (Default value = None)
        rs_kw_1: dict :
             (Default value = None)
        patch_kw_1:dict :
             (Default value = None)
        highlight_set_1: tuple :
             (Default value = None)
        lbl_kw_2:dict :
             (Default value = None)
        ledge_trim_2:int :
             (Default value = None)
        rs_kw_2:dict :
             (Default value = None)
        patch_kw_2:dict :
             (Default value = None)
        highlight_set_2:tuple :
             (Default value = None)

        Returns
        -------

        """
        
        fig = plt.figure(figsize=figsize)
        
        # setup
        genes_1 = self.ledge_1['gene'].values
        genes_2 = self.ledge_2['gene'].values
              
        # DEFAULTS, Patch will always follow running sum color
        rs_prop_1 = {'color':'#AC3220'} ; patch_prop_1 =  rs_prop_1.copy() # Chinese red
        if rs_kw_1 is not None:
            rs_prop_1.update(rs_kw_1)
        if patch_kw_1 is not None:
            patch_prop_1.update(patch_kw_1)
            
        rs_prop_2 = {'color':'#50808E'} ; patch_prop_2 = rs_prop_2.copy() # Teal Blue 
        if rs_kw_2 is not None:
            rs_prop_2.update(rs_kw_2)
        if patch_kw_2 is not None:
            patch_prop_2.update(patch_kw_2)
    
        lbl_prop_1 = {'fontsize':4, 'rotation':90, 'ha':'center', 'va':'center'}
        lbl_prop_2 = lbl_prop_1.copy()
        if lbl_kw_1 is not None:
            lbl_prop_1.update(lbl_kw_1)
        if lbl_kw_2 is not None:
            lbl_prop_2.update(lbl_kw_2)
            
        stat_prop = {'loc':3, "title":self.regulator} # Stats go to bottom left by default
        if stat_kw is not None:
            stat_prop.update(stat_kw)
      
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 4, 1], hspace=0.1)
        
        # First gene set leading edge, above
        ax1 = fig.add_subplot(gs[0])          
        pl._plot_ledge_labels(self.ledge_xinfo_1, genes=genes_1, ax=ax1, 
                              highlight=highlight_set_1, trim_ledge=ledge_trim_1, **lbl_prop_1)
        
        # Running sums
        ax2 = fig.add_subplot(gs[1])
        if rs_kw_1 is not None:
            rs_prop_1.update(rs_kw_1)
        if rs_kw_2 is not None:
            rs_prop_2.update(rs_kw_2)
            
        pl._plot_run_sum(self.rs_1, self.es_idx_1, ax=ax2, **rs_prop_1)
        pl._plot_run_sum(self.rs_2, self.es_idx_2, ax=ax2, add=True, **rs_prop_2)
        leg = pl._add_reg_legend(rs_prop_1.get('color'), rs_prop_2.get('color'), leg_kw=leg_kw)
        ax2.add_artist(leg)    
        stats = pl._stats_legend(self.aREA_nes, self.pval, leg_kw=stat_prop)
        ax2.add_artist(stats)
        ax2.set_xticks([])
        
        # 
        ax3 = fig.add_subplot(gs[2])   
        pl._plot_ledge_labels(self.ledge_xinfo_2, genes=genes_2, ax=ax3, upper=False, 
                              highlight=highlight_set_2, trim_ledge=ledge_trim_2, **lbl_prop_2)
        
        # Zooming
        pl.zoom_effect(ax1, ax2, patch_kw=patch_prop_1)
        pl.zoom_effect(ax3, ax2, upper=False, patch_kw=patch_prop_2)    
        
        return fig 
    
    
class GseaRegTMultSigs(Gsea1TMultSigs, GseaReg):
    """ """
     
    def __init__(self, 
                 dset: pd.DataFrame, 
                 regulon: pd.DataFrame,  
                 ordered: bool=True):
                    
        if not isinstance(dset, pd.DataFrame):
            raise TypeError('Need an indexed pandas DataFrame, please.')
        
        else: 
            self.dset = _prep_ges(dset)
            self.ns = len(self.dset)
            self.samples = self.dset.columns.values
             
        if not isinstance(regulon, pd.DataFrame):
            raise TypeError('Regulon needs to be supplied as a DataFrame, please.')
        else:
            self.regulator = ''.join(set(regulon['source']))
            self.regulon_in = regulon
       
        # if not all(self.reg_df_in.columns.values == np.array(['target', 'mor', 'likelihood'])):
        #     raise ValueError('Regulon value DataFrame must have columns: target, mor, likelihood')

        if not np.in1d(self.regulon_in['target'].values, self.dset.index).any():
            raise ValueError("None of the regulator targets were found in the gene expression signature index!")
        else:
            self.reg_df, self.reg_pos, self.reg_neg = self._split_regulon(self.regulon_in, self.dset.index)
            
        self.stats = self._get_stats(dset=self.dset, 
                                     regulon=regulon, 
                                     minsize=len(self.reg_df),
                                     samples=self.samples)
        
        self.gs_idx = self._find_hits(self.dset, self.reg_pos['target'].to_list(), 
                                      self.reg_neg['target'].to_list())

        if ordered:
            idx = self.stats['NES'].argsort().values
            self.stats = self.stats.take(idx, axis=0)
            self.stats.reset_index(inplace=True, drop=True)
            self.gs_idx = self.gs_idx.take(idx, axis=0)
            
    
    def __repr__(self):
        
        """String representation for the Gsea1TMultSig class"""

        return (
                f"GseaRegMultSig(Number of genes: {self.ns}\n"
                f"Number of signatures: {len(self.samples)}\n"
                f"Regulator: {self.regulator}\n"
                f"Number targets: {len(self.reg_df_in)}, Overlap: {len(self.reg_df)}\n"
                )        
        
    def plot(self, 
             figsize: tuple = (3, 3),
             regulon_colors:tuple = ('#AC3220', '#50808E'),
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
        regulon_colors:tuple :
             (Default value = ('#AC3220')
        '#50808E') :
            
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
        
        evt_data = [arr for tup in self.gs_idx.values for arr in tup]
        
        norm_prop = {'vmin':np.min(df['NES'].values),
                     'vcenter':0,  
                    'vmax':np.max(df['NES'].values)}
        
        evt_prop = {'linelengths':0.33, 'alpha':0.7, 'linewidths':0.5}
        
        pcm_prop = {'cmap':'PuOr_r', 'edgecolor':'.25', 'lw':.5}
        
        # Prettify FDR format
        df['FDR'] = df['FDR'].apply(pl._fdr_formatter)
                
        # In this first bit, we determine the NES color scheme
        if norm_kw is not None:
            norm_prop.update(norm_kw)
            
        norm = mcolors.TwoSlopeNorm(**norm_prop)
        norm_map = norm(df['NES'].values)
        
        # TODO: Pretty heuristically, maybe improve in the future
        df['color'] = pd.cut(norm_map, 
                             bins=[0, 0.15, 0.85, 1], 
                             ordered=False,
                             labels=['w','k','w'], 
                             include_lowest=True)
        
        fig = plt.figure(figsize=figsize) 
        
        # height ratio may change according to figsize and number of signatures,
        # carve out a fixed number of inches. 0.15 translates to roughly 0.38 cm. 
        height_clb = 0.15
        rest = figsize[1]-height_clb
                
        gs = fig.add_gridspec(nrows=2, 
                      ncols=2, 
                      hspace=0.2, 
                      wspace=0.025,
                      width_ratios=[10,1],
                      height_ratios=[height_clb, rest])
        
        # Plot 1: Illustrate the ordered signature as a colorbar
        ax1 = fig.add_subplot(gs[0])
        cb = fig.colorbar(ScalarMappable(cmap=coolwarm), 
                          orientation='horizontal', ticks=[], cax=ax1)
        cb.outline.set_visible(False)
        ax1.text(0, 0.5, 'Low', color='w', ha='left', va='center', fontsize='x-small')
        ax1.text(1, 0.5, 'High', color='w', ha='right', va='center', fontsize='x-small')
    
        # Plot 2: Illustrate targets
        rc_new = {'xtick.bottom':False, 'xtick.labelbottom':False,
                      'xtick.top':True, 'xtick.labeltop':True, 
                      'xtick.labelsize':'x-small', 
                      'xtick.major.pad':1.0}
        
        if evt_kw is not None:
            evt_prop.update(evt_kw)
        
        lofs, ytick_pos = pl._prepare_multi_gseareg(linelengths=evt_prop.get('linelengths'), 
                                         number_of_ys=len(self.samples))
        
        cls = np.tile(regulon_colors, len(self.samples))
            
        with plt.rc_context(rc_new):
            ax2 = fig.add_subplot(gs[1,0])                  
            ax2.eventplot(evt_data, lineoffsets=lofs, colors=cls, **evt_prop)
            ax2.set_yticks(ytick_pos)
            add_y = evt_prop.get('linelengths')
            ax2.set_ylim([0-add_y, np.max(lofs)+add_y])
            ax2.set_yticklabels(df['signature_name'].values, fontsize='x-small', va='top')
            ax2.set_xlim([-.5, self.ns+.5])
            pl._format_xaxis_ges(self.ns, ax=ax2)
        # x-axis formating
            for spine in ['right', 'left']:
                ax2.spines[spine].set_visible(False)
            ax2.annotate(text=f"{self.regulator} regulon", 
                         xy=(0.5, -0.01), xycoords='axes fraction', 
                         ha='center', va='top', fontsize='small')
    
        # Plot 3: NES heatmap
        ax3 = fig.add_subplot(gs[1,1])
        if pcm_kw is not None:
            pcm_prop.update(pcm_kw)
            
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
        ax4.text(0.5, 0, 'NES', fontsize='x-small', c='k', ha='center', va='top')
        ax4.text(1.1, 0, 'FDR', fontsize='x-small', c='k', ha='left', va='top')
        ax4.axis('off')
        
        return fig




class GseaMultReg:
    
    """To implement one-tailed gene set enrichment of multiple regulons on one gene expression signature
    and visualize the results

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, 
                 ges: pd.Series, 
                 regulons: pd.DataFrame,
                 minsize: int = 20,
                 ordered: bool=True):
        
        if not isinstance(ges, pd.Series):
            raise TypeError('Need an indexed pandas Series, please.')
        else:
            self.ges = _prep_ges(ges)
            self.ns = len(self.ges)
            self.along_scores = [*range(self.ns)]
            self.minsize = minsize
            
        if not isinstance(regulons, pd.DataFrame):
            raise TypeError('Regulons need to be supplied as a DataFrame, please.')
        
        self.regulons_in = regulons['source'].nunique()
           
        self.targets = self._split_regulons(regulons=regulons)
        
        self.target_idx = self._find_hits(self.ges, self.targets)
        
        self.stats = self._get_stats(self.ges, regulons, minsize=self.minsize)
                  
        if ordered:
            idx = self.stats['NES'].argsort().values
            self.stats = self.stats.take(idx, axis=0)
            self.stats.reset_index(inplace=True, drop=True)
            
            
    def __repr__(self):

        """String representation for the Gsea1TMultSet class"""

        return (
                f"GseaMultReg(GES length: {self.ns}\n"
                f"Regulators provided: {self.regulons_in}\n"
                f"Regulators evaluated: {len(self.stats)}"
                )
        
    def _split_targets(self, df):
        """

        Parameters
        ----------
        df :
            

        Returns
        -------

        """
        mask = df['mor']>=0
        return df['target'].values[mask], df['target'].values[~mask]
  
    def _split_regulons(self, regulons:pd.DataFrame):
        """

        Parameters
        ----------
        regulons:pd.DataFrame :
            

        Returns
        -------

        """
        targets = regulons.groupby('source').apply(self._split_targets)
        
        target_df = pd.DataFrame(zip(*targets), 
                               columns=targets.index, 
                               index=['positive_targets', 'negative_targets'])
        
        return target_df.T
        
    def _find_hits(self, 
                   ges: pd.Series, 
                   target_df: pd.DataFrame):
        """

        Parameters
        ----------
        ges :
            pd.Series:
        gene_sets :
            dict:
        ges: pd.Series :
            
        target_df: pd.DataFrame :
            

        Returns
        -------

        """

        # This will have to be sorted first, which is the case in this Class
        ges = ges.argsort()
        
        pos = [ges[ges.index.intersection(val)].values for val in target_df['positive_targets'].values]
        neg = [ges[ges.index.intersection(val)].values for val in target_df['negative_targets'].values]
        
        pos_df = pd.DataFrame(zip(pos, neg), 
                            index=target_df.index, 
                            columns=['pos_idx', 'neg_idx'])
        
        return pos_df
    
    def _get_stats(self, 
                   ges: pd.Series,
                   regulons: pd.DataFrame,
                   minsize:int):
        
        """Computes normalized enrichment scores and some tools for later visualization

        Parameters
        ----------
        ges :
            pd.Series:
        regulon :
            dict:
        ges: pd.Series :
            
        regulons: pd.DataFrame :
            
        minsize:int :
            

        Returns
        -------

        """
        
        nes = aREA(ges, regulons, minsize=minsize)
        nes.columns = ['NES']
        nes.index.name = 'Regulator'
        nes['pvals'] = norm.sf(np.abs(nes['NES'].values))*2 # retrieve two-sided p-value from normal distribution
        nes['FDR'] = multipletests(nes['pvals'].values, method = 'fdr_bh')[1] #FDR
        nes.reset_index(inplace=True)

        return nes
                    
    def plot(self, 
             figsize: tuple = (3, 3),
             conditions: tuple = ('A', 'B'),
             regulon_colors: tuple=('#AC3220', '#50808E'),
             ges_symlog: bool=True,
             ges_stat_fmt:str='1.0f',
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
        regulon_colors: tuple :
             (Default value = ('#AC3220')
        '#50808E') :
            
        ges_symlog: bool :
             (Default value = True)
        ges_stat_fmt:str :
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
        evt_prop = {'alpha':0.7, 'linelengths':0.5, 'linewidths':0.5}         
        norm_prop = {'vcenter':0, 
                    'vmin':np.min(df['NES'].values), 
                    'vmax':np.max(df['NES'].values)}
        pcm_prop = {'cmap':'PuOr_r', 'edgecolor':'.25', 'lw':.5}
        
        if norm_kw is not None:
            norm_prop.update(norm_kw)
             
        norm = mcolors.TwoSlopeNorm(**norm_prop)
        norm_map = norm(df['NES'].values)
        
        
         # TODO: this works fairly well for now, but could be improved eventually
        df['color'] = pd.cut(norm_map, 
                             bins=[0, 0.15, 0.85, 1], 
                             ordered=False,
                             labels=['w','k','w'], 
                             include_lowest=True)
        
                    
        if subset is not None:
            df = self._filter_multset_stats(df, subset)
            df.reset_index(inplace=True)
        else:
            top_rp = df.take(df['NES'].abs().nlargest(20).index)['Regulator']
            df = df[df['Regulator'].isin(top_rp)]
            df.reset_index(inplace=True, drop=True)
            
        targets = self.target_idx.loc[df['Regulator']].values
        evt_data = [arr for tup in targets for arr in tup]
                   
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
        pl._plot_ges(self.along_scores, 
                     self.ges.values, 
                     ges_type=ges_type, 
                     conditions=conditions, 
                     is_high_to_low=False, 
                     symlog=ges_symlog,
                     stat_fmt=ges_stat_fmt,
                     ax=ax1, **ges_prop)
        
        # Plot 2: Illustrate targets
        ax2 = fig.add_subplot(gs[1,0])
        
        lofs, ytick_pos = pl._prepare_multi_gseareg(linelengths=evt_prop.get('linelengths'), 
                                         number_of_ys=len(df))
        
        
        cls = np.tile(regulon_colors, len(df))
        
        if evt_kw is not None:
            evt_prop.update(evt_kw)
       
        
        ax2.eventplot(evt_data, lineoffsets=lofs, colors=cls, **evt_prop)
        ax2.set_yticks(ytick_pos)
        add_y = evt_prop.get('linelengths')
        ax2.set_ylim([0-add_y, np.max(lofs)+add_y])
        ax2.set_yticklabels(df['Regulator'].values, fontsize='x-small', va='top')
        ax2.set_xlim([-.5, self.ns+.5])
        pl._format_xaxis_ges(self.ns, ax=ax2)
              
        # remove spines
        for spine in ['right', 'left', 'top']:
            ax2.spines[spine].set_visible(False)

        # Plot 3: NES heatmap            
        ax3 = fig.add_subplot(gs[1,1])
        
        if pcm_kw is not None:
            pcm_prop.update(pcm_kw)
        
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

