
# documentation
from typing import Union

# as always
import numpy as np
import pandas as pd

# Plotting utils
from matplotlib import pyplot as plt
from . import plotting as pl

# gene set enrichment and helpers
from .aREA import aREA
from .utils import gene_sets_to_regulon, _prep_ges

# Stats
from statsmodels.stats.multitest import multipletests
from scipy.cluster import hierarchy
from scipy.spatial import distance


class Viper:
    
    """Class for tranforming gene expression matrices to 
    pathway or regulatory gene activity matrices"""
    
    def __init__(self, 
                 dset: pd.DataFrame, 
                 regulons: Union[dict, pd.DataFrame], 
                 dset_filter: bool = False,
                 minsize: int = 20):
        
        self.minsize = minsize
        self.dset_filter = dset_filter
        self.dset = _prep_ges(dset)
        self.ns = len(self.dset)
        self.samples = self.dset.columns.values
        
        if isinstance(regulons, dict):
            self.regulon = gene_sets_to_regulon(regulons, minsize=self.minsize)
        elif isinstance(regulons, pd.DataFrame):
            self.regulon = regulons
        else:
            raise ValueError('Regulon in wrong format')
        
        self.nregs = self.regulon['source'].nunique()
        
        self.nes = aREA(self.dset, 
                        self.regulon, 
                        self.minsize, 
                        self.dset_filter)
        
    def __repr__(self):
        
        """String representation for the Viper class"""

        return (
                f"Viper(Number of genes: {self.ns}\n"
                f"Number of signatures: {len(self.samples)}\n"
                f"Regulators/Pathways: n={len(self.nregs)}\n"
            )    
    
    
    def plot(self, 
            cluster_rows: bool = False,
            show_gene_labels:bool=False,
            show_numbers:bool = False,
            number_fmt:str = '1.1f',
            figsize: tuple=None,
            number_kw:dict = None,
            pcm_kw:dict = None):
        
        ndata = self.nes.copy()
        nrows, ncols = ndata.shape
                          
        pcm_prop = {'cmap':plt.cm.RdBu_r, 'edgecolors':'k', 'linewidths':0.25}
        if pcm_kw is not None:
            pcm_prop.update(pcm_kw)
            
        if cluster_rows:
            cmat = ndata.T.corr()
            cmat_condensed = distance.squareform(1 - cmat)
            zvar = hierarchy.linkage(cmat_condensed, method='complete', optimal_ordering=True)
            dn = hierarchy.dendrogram(zvar, labels=ndata.index, orientation='left', no_plot=True)
            dn_max = np.max([y for x in dn['dcoord'] for y in x])          
        
        if figsize is None:
            height = nrows * 0.15
            width_base = ncols * 0.3
            if cluster_rows:
                width = width_base + 1/8 * width_base
            else:
                width = width_base
            figsize = (width, height)
        
        if cluster_rows:
            
            fig, (ax1, ax2) = plt.subplots(ncols=2, 
                                           figsize=figsize, 
                                           sharex=False,
                                           gridspec_kw={'width_ratios':[1, 8],
                                                        'wspace':0.02})

            # Draw dendrogram
            for xline, yline in zip(dn['icoord'], dn['dcoord']):
                ax1.plot(yline, xline, c='k', lw=0.8)
                ax1.set_xlim(dn_max, 0)
                ax1.axis('off')
            
            # Draw heatmap
            # reorder data
            ndata = ndata.loc[dn['ivl']]

            mesh = ax2.pcolormesh(ndata.values, **pcm_prop)
        
            if show_gene_labels:
                ax2.set_yticks(np.arange(nrows) + 0.5)
                # TODO: Add more flexibility here: trimming names
                ax2.set_yticklabels(ndata.index.str.split('_', n=1).str[-1], fontsize='x-small')
                ax2.yaxis.set_label_position('right')
                ax2.yaxis.tick_right()
            else:
                ax2.set_yticks([])
            
            ax2.set_xticks(np.arange(len(self.samples)) + 0.5)
            ax2.set_xticklabels(self.samples, fontsize='x-small', 
                                ha="right", rotation=45, rotation_mode='anchor')
            ax2.set_ylabel('') 
                                    
            # TODO: colorbar adjustments
            cbar_height = 0.02 * 50/nrows
            cbar_width = 0.4 * 5/ncols
            cax = ax2.inset_axes([0.04, 1.01, cbar_width, cbar_height], transform=ax2.transAxes)
            cb = plt.colorbar(mesh, ax=ax2, cax=cax, orientation='horizontal')
            cb.outline.set_visible(False)
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.tick_params(labelsize='xx-small')
            cb.ax.annotate(text='NES', xy=(1.02, 0.25), xycoords='axes fraction', fontsize='x-small')

            if show_numbers:
                mesh.update_scalarmappable()
                height, width = self.ns, len(self.samples)
                xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
                
                number_prop = {'ha':"center", 'va':"center", 'fontsize':'xx-small'}
                if number_kw is not None:
                    number_prop(number_kw)
        
                for x, y, rgba_in, val in zip(xpos.flat, 
                                                ypos.flat, 
                                                mesh.get_facecolors(), 
                                                ndata.values.flat):
                    text_color = pl._color_light_or_dark(rgba_in)
                    annot = f'{val:{number_fmt}}' 
                    ax2.text(x, y, annot, color=text_color, **number_prop)
                    
        else:
            
            fig, ax = plt.subplots(figsize=figsize)
            # Draw heatmap
            mesh = ax.pcolormesh(ndata.values, **pcm_prop)
            # TODO: Currently empirical, maybe need to give to user
             
            if show_gene_labels:
                ax.set_yticks(np.arange(nrows) + 0.5)
                # TODO: Add more flexibility here: trimming names
                ax.set_yticklabels(ndata.index.str.split('_', n=1).str[-1], fontsize='x-small')
                ax.yaxis.set_label_position('right')
                ax.yaxis.tick_right()
            else:
                ax.set_yticks([])
            
            ax.set_xticks(np.arange(len(self.samples)) + 0.5)
            ax.set_xticklabels(self.samples, fontsize='x-small', 
                                ha="right", rotation=45, rotation_mode='anchor')
            ax.set_ylabel('') 
                                    
            # TODO: colorbar adjustments
            cax = ax.inset_axes([0.04, 1.01, 0.6, 0.015], transform=ax.transAxes)
            cb = plt.colorbar(mesh, ax=ax, cax=cax, orientation='horizontal')
            cb.outline.set_visible(False)
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.tick_params(labelsize='xx-small')

            if show_numbers:
                mesh.update_scalarmappable()
                height, width = self.ns, len(self.samples)
                xpos, ypos = np.meshgrid(np.arange(width) + .5, np.arange(height) + .5)
                
                number_prop = {'ha':"center", 'va':"center", 'fontsize':'xx-small'}
                if number_kw is not None:
                    number_prop(number_kw)
        
                for x, y, rgba_in, val in zip(xpos.flat, 
                                                ypos.flat, 
                                                mesh.get_facecolors(), 
                                                ndata.values.flat):
                    text_color = pl._color_light_or_dark(rgba_in)
                    annot = f'{val:{number_fmt}}' 
                    ax.text(x, y, annot, color=text_color, **number_prop)
                    
                
        return fig
                    