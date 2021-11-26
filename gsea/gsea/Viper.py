
# documentation
from typing import Union
import matplotlib

# as always
import numpy as np
import pandas as pd

# Plotting utils
from matplotlib import pyplot as plt
from . import plotting as pl
from matplotlib.colors import CenteredNorm

# gene set enrichment and helpers
from .aREA import aREA
from .utils import gene_sets_to_regulon, _prep_ges

# Stats
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
        self.ngenes = len(self.dset)
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
                f"Viper(Number of genes: {self.ngenes}\n"
                f"Number of signatures: {len(self.samples)}\n"
                f"Regulators/Pathways: n={self.nregs}\n"
            )    
        
    
    def _draw_dendrogram(self, 
                         dendro,
                         type:str, 
                         ax=None):
        
        if ax is None:
            ax = plt.gca()
            
        number_of_leaves = len(dendro['leaves'])
        max_dependent_coord = max(map(max, dendro['dcoord']))
            
        if type == 'row':
            
            [ax.plot(yline, xline, c='k', lw=0.8) for xline, yline in zip(dendro['icoord'], dendro['dcoord'])]
            ax.set_ylim(0, number_of_leaves * 10)
            ax.set_xlim(max_dependent_coord, 0)
        elif type == 'col':
            [ax.plot(xline, yline, c='k', lw=0.8) for xline, yline in zip(dendro['icoord'], dendro['dcoord'])]
            ax.set_xlim(0, number_of_leaves * 10)
            ax.set_ylim(max_dependent_coord, 0)
        else:
            print(f'Do not know this option: {type}!')
        ax.axis('off')
        
    def _add_numbers(self, 
                     mesh, 
                     ndata:pd.DataFrame, 
                     number_fmt: str,
                     number_kw:dict,
                     ax=None):
        
        if ax is None:
            ax = plt.gca()
            
        mesh.update_scalarmappable()
        height, width = ndata.shape
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

    
    def plot(self, 
            cluster_rows: bool = False,
            cluster_cols: bool = False,
            show_row_labels:bool=False,
            show_col_labels: bool=True,
            show_numbers:bool = False,
            number_fmt:str = '1.1f',
            figsize: tuple=None,
            number_kw:dict = None,
            pcm_kw:dict = None,
            norm_kw:dict = None):
        
        ndata = self.nes.copy()
        nrows, ncols = ndata.shape
        
        if cluster_rows and ncols<=2:
            print('Ignoring row clustering with only 2 columns!')
            cluster_rows = False 
        
        if cluster_cols and nrows<=2:
            print('Ignoring column clustering with only 2 rows!')
            cluster_cols = False 
          
        pcm_prop = {'cmap':plt.cm.RdBu_r, 'edgecolors':'k', 'linewidths':0.15}
        if pcm_kw is not None:
            pcm_prop.update(pcm_kw)
            
        norm_prop = {'vcenter':0}
        if norm_kw is not None:
            norm_prop.update(norm_kw)
            
        norm = CenteredNorm(**norm_prop)
            
        if cluster_cols:
            zvar_col = hierarchy.linkage(ndata.T, method='complete', metric='euclidean')
            dn_col = hierarchy.dendrogram(zvar_col, labels=ndata.columns, orientation='bottom', no_plot=True)   
        
        if cluster_rows:         
            zvar_row = hierarchy.linkage(ndata, method='complete', metric='euclidean')
            dn_row = hierarchy.dendrogram(zvar_row, labels=ndata.index, orientation='left', no_plot=True)
           
        if cluster_rows and cluster_cols:            
            if figsize is None:
                height = nrows * 0.2
                width = ncols * 0.3
            else:
                width, height = figsize
            
            dendro_size_row = 0.15
            dendro_size_col = 0.3
            cbar_height = 0.3
            height_rest = height-dendro_size_col-cbar_height
            hspace = 0.15/height
            width_rest = width-dendro_size_row
            wspace = 0.15/width             
                
            fig = plt.figure(figsize=(width, height), dpi=150)
            
            gs = fig.add_gridspec(nrows=3, ncols=2, 
                                  width_ratios=[dendro_size_row,width_rest], wspace=wspace, 
                                  height_ratios=[height_rest, dendro_size_col, cbar_height], hspace=hspace)
            
            # Draw row dendrogram
            ax_row_dn = fig.add_subplot(gs[0,0])
            self._draw_dendrogram(dn_row, type='row', ax=ax_row_dn)
            # TODO: FIX DENDROGRAM ADJUSTMENTS
            ax_row_dn.set_ylim(0, len(dn_row['leaves']) * 10)
            
            # Draw heatmap
            # reorder data
            ndata = ndata.iloc[dn_row['leaves'], dn_col['leaves']]
            
            ax_mesh = fig.add_subplot(gs[0,1])
            mesh = ax_mesh.pcolormesh(ndata.values, norm=norm, **pcm_prop)
        
            if show_row_labels:
                ax_mesh.set_yticks(np.arange(nrows) + 0.5)
                # TODO: Add more flexibility here: trimming names
                ax_mesh.set_yticklabels(ndata.index.str.split('_', n=1).str[-1], fontsize='x-small')
                ax_mesh.yaxis.set_label_position('right')
                ax_mesh.yaxis.tick_right()
            else:
                ax_mesh.set_yticks([])
            ax_mesh.set_ylabel('') 
            
            if show_col_labels:
                ax_mesh.set_xticks(np.arange(ncols) + 0.5)
                ax_mesh.xaxis.tick_top()
                ax_mesh.xaxis.set_label_position('top')       
                ax_mesh.set_xticklabels(ndata.columns, fontsize='x-small', 
                                    ha="left", rotation=45, rotation_mode='anchor')
            else: 
                ax_mesh.set_xticks([])

            if show_numbers:
                self._add_numbers(mesh, ndata, number_fmt, number_kw, ax_mesh)
                      
            ax_col_dn = fig.add_subplot(gs[1,1])
            self._draw_dendrogram(dn_col, type='col', ax=ax_col_dn)
            
            ax_cbar = fig.add_subplot(gs[2, 1])
            ax_cbar.axis('off')
            # TODO: colorbar adjustments
            cb = fig.colorbar(mesh, ax=ax_cbar, fraction=0.8, orientation='horizontal', shrink=0.8, aspect=10)
            # cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize='xx-small')
          
        
        elif cluster_rows:
            
            if figsize is None:
                height = nrows * 0.2
                width = ncols * 0.3
            else:
                width, height = figsize
            
            dendro_size_row = 0.15
            cbar_height = 0.3
            height_rest = height-cbar_height
            hspace = 0.15/height
            width_rest = width-dendro_size_row
            wspace = 0.15/width   
                
            fig = plt.figure(figsize=(width, height), dpi=150)
            
            gs = fig.add_gridspec(nrows=2, ncols=2, 
                                  width_ratios=[dendro_size_row, width_rest], 
                                  wspace=wspace,
                                  height_ratios=[cbar_height, height_rest],
                                  hspace=hspace)
          
            # reorder data
            ndata = ndata.iloc[dn_row['leaves']]
            ax_mesh = fig.add_subplot(gs[1,1])
            mesh = ax_mesh.pcolormesh(ndata.values, **pcm_prop)
        
            if show_row_labels:
                ax_mesh.set_yticks(np.arange(nrows) + 0.5)
                # TODO: Add more flexibility here: trimming names
                ax_mesh.set_yticklabels(ndata.index.str.split('_', n=1).str[-1], fontsize='x-small')
                ax_mesh.yaxis.set_label_position('right')
                ax_mesh.yaxis.tick_right()
            else:
                ax_mesh.set_yticks([])
            ax_mesh.set_ylabel('') 
            
            if show_col_labels:      
                ax_mesh.set_xticks(np.arange(ncols) + 0.5)      
                ax_mesh.set_xticklabels(ndata.columns, fontsize='x-small', 
                                        ha="right", rotation=45, rotation_mode='anchor')
            else:
                ax_mesh.set_xticks([])
                                               
            if show_numbers:
                self._add_numbers(mesh, ndata, number_fmt, number_kw, ax_mesh)
                
            ax_cbar = fig.add_subplot(gs[0, 1])
            ax_cbar.axis('off')
            # TODO: colorbar adjustments
            cb = fig.colorbar(mesh, ax=ax_cbar, fraction=0.8, orientation='horizontal', shrink=0.8, aspect=10)
            # cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize='xx-small')
            
                # Draw row dendrogram
            ax_row_dn = fig.add_subplot(gs[1,0])
            
            self._draw_dendrogram(dn_row, type='row', ax=ax_row_dn)
        
                     
        elif cluster_cols:
            
            if figsize is None:
                height = nrows * 0.2
                width = ncols * 0.3
            else:
                width, height = figsize
            
            dendro_size_col = 0.3
            cbar_height = 0.3
            height_rest = height-dendro_size_col-cbar_height
            hspace = 0.15/height
                
            fig = plt.figure(figsize=(width, height), dpi=150)
            
            gs = fig.add_gridspec(nrows=3, ncols=1, 
                                  height_ratios=[height_rest, dendro_size_col, cbar_height], 
                                  hspace=hspace)
            
            # Draw heatmap
            # reorder data
            ndata = ndata.iloc[:,dn_col['leaves']]
            
            ax_mesh = fig.add_subplot(gs[0])
            mesh = ax_mesh.pcolormesh(ndata.values, **pcm_prop)
        
            if show_row_labels:
                ax_mesh.set_yticks(np.arange(nrows) + 0.5)
                # TODO: Add more flexibility here: trimming names
                ax_mesh.set_yticklabels(ndata.index.str.split('_', n=1).str[-1], fontsize='x-small')
                ax_mesh.yaxis.set_label_position('right')
                ax_mesh.yaxis.tick_right()
            else:
                ax_mesh.set_yticks([])
                      
            if show_col_labels:      
                ax_mesh.set_xticks(np.arange(ncols) + 0.5)
                ax_mesh.xaxis.tick_top()
                ax_mesh.xaxis.set_label_position('top')       
                ax_mesh.set_xticklabels(ndata.columns, fontsize='x-small', 
                                        ha="left", rotation=45, rotation_mode='anchor')
            else:
                ax_mesh.set_xticks([])
       
            if show_numbers:
                self._add_numbers(mesh, ndata, number_fmt, number_kw, ax_mesh)
                      
            ax_col_dn = fig.add_subplot(gs[1])
            self._draw_dendrogram(dn_col, type='col', ax=ax_col_dn)
            
            ax_cbar = fig.add_subplot(gs[2])
            ax_cbar.axis('off')
            # TODO: colorbar adjustments
            cb = fig.colorbar(mesh, ax=ax_cbar, fraction=0.8, orientation='horizontal', shrink=0.8, aspect=10)
            # cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize='xx-small')
        
        else:
            if figsize is None:
                height = nrows * 0.2
                width = ncols * 0.3
            else:
                width, height = figsize
        
            cbar_height = 0.3
            height_rest = height-cbar_height
            hspace = 0.15/height
                
            fig = plt.figure(figsize=(width, height), dpi=150)
            
            gs = fig.add_gridspec(nrows=2, ncols=1, 
                                  height_ratios=[cbar_height, height_rest], 
                                  hspace=hspace)
            # Draw heatmap
            
            ax_mesh = fig.add_subplot(gs[1])
            mesh = ax_mesh.pcolormesh(ndata.values, **pcm_prop)
             
            if show_row_labels:
                ax_mesh.set_yticks(np.arange(nrows) + 0.5)
                # TODO: Add more flexibility here: trimming names
                ax_mesh.set_yticklabels(ndata.index.str.split('_', n=1).str[-1], fontsize='x-small')
                ax_mesh.yaxis.set_label_position('right')
                ax_mesh.yaxis.tick_right()
            else:
                ax_mesh.set_yticks([])
            
            if show_col_labels:
                ax_mesh.set_xticks(np.arange(ncols) + 0.5)
                ax_mesh.set_xticklabels(ndata.columns, fontsize='x-small', 
                                ha="right", rotation=45, rotation_mode='anchor')
            else:
                ax_mesh.set_xticks([])
                                
            if show_numbers:
                self._add_numbers(mesh, ndata, number_fmt, number_kw, ax_mesh)
                
            ax_cbar = fig.add_subplot(gs[0])
            ax_cbar.axis('off')
            # TODO: colorbar adjustments
            cb = fig.colorbar(mesh, ax=ax_cbar, fraction=0.8, orientation='horizontal', shrink=0.8, aspect=10)
            # cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize='xx-small')
                
        return fig
                    