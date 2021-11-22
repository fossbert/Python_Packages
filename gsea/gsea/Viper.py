
# documentation
from typing import Union

# as always
import numpy as np
import pandas as pd

# Plotting utils
from matplotlib import pyplot as plt
from pandas.core.algorithms import isin
from . import plotting as pl

# gene set enrichment and helpers
from .aREA import aREA
from .utils import gene_sets_to_regulon, _prep_ges

# Stats
from statsmodels.stats.multitest import multipletests


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
            figsize: tuple=(6,8),
            show_gene_labels:bool=False,
            show_numbers:bool = False,
            number_fmt:str = '1.1f',
            number_kw:dict = None,
            show_varlabels: bool = False,
            pcm_kw:dict = None):
        
        ndata = self.nes.copy()
        
        if figsize is None:
            height = round(self.ns * 0.16)
            
            if show_gene_labels:
                annot_width = np.max(ndata.index.str.len)/20
                sample_width = round(len(self.samples) * 0.16 * 3/2)
                width = annot_width + sample_width
            else:
                width = round(len(self.samples) * 0.16 * 3/2)
            
            figsize = (width, height)
        
        fig, ax = plt.subplots(figsize=figsize)
        
      
                    
        pcm_prop = {'cmap':plt.cm.RdBu_r, 'edgecolors':'k', 'linewidths':0.25}
        if pcm_kw is not None:
            pcm_prop.update(pcm_kw)

        # Draw heatmap
        mesh = ax.pcolormesh(ndata.values, **pcm_prop)
        # TODO: Currently empirical, maybe need to give to user
        ax.set_aspect(2/3)
        
        if show_gene_labels:
            ax.set_yticks(np.arange(len(ndata)) + 0.5)
            # TODO: Add more flexibility here: trimming names
            ax.set_yticklabels(ndata.index.str.split('_', n=1).str[-1], fontsize='x-small')
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
                ax.text(x, y, annot, color=text_color, **number_prop)
                
                
        return fig
                    