
"""Outsourced some helper functions for readability"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from typing import Union




pyrea_rc_params



def _plot_ges(along_scores: list, 
              ges_values:np.ndarray, 
              ges_type: str, 
              conditions: tuple, 
              symlog:bool,
              stat_fmt:str,
              is_high_to_low: bool = True,
              ax = None,
              **kwargs):
    """

    Parameters
    ----------
    along_scores: list :
        
    ges_values:np.ndarray :
        
    ges_type: str :
        
    conditions: tuple :
        
    symlog:bool :
        
    stat_fmt:str :
        
    is_high_to_low: bool :
         (Default value = True)
    ax :
         (Default value = None)
    **kwargs :
        

    Returns
    -------

    """
    if ax is None:
        ax = plt.gca()
        
    ax.fill_between(along_scores, ges_values, **kwargs)
    ax.set_xticks([])
    if ges_type is None:
        ges_type = 'Gene score'
    ax.set_ylabel(ges_type, fontsize='xx-small')
    ax.axhline(y=0, linestyle='-', lw=0.5, c='.15')
    ax.set_xlim([-.5, len(ges_values)+.5])
    p1, p2 = conditions
    if is_high_to_low:
        ax.annotate(p1, xy=(0.05, 0.05), xycoords='axes fraction', ha='left', va='bottom', fontsize='small')
        ax.annotate(p2, xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top', fontsize='small')
    else:
        ax.annotate(p1, xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize='small')
        ax.annotate(p2, xy=(0.95, 0.05), xycoords='axes fraction', ha='right', va='bottom', fontsize='small')
    ax.yaxis.set_tick_params(labelsize='xx-small')
    if symlog:
        ax.set_yscale('symlog')
    yticks = [np.min(ges_values), 0, np.max(ges_values)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(list(map(lambda x: f'{x:{stat_fmt}}', yticks)))
    
    return ax

def _plot_run_sum(run_sum:np.ndarray, 
                  es_idx: int, 
                  add: bool=False,
                  ax=None,
                  **kwargs):
    """

    Parameters
    ----------
    run_sum:np.ndarray :
        
    es_idx: int :
        
    add: bool :
         (Default value = False)
    ax :
         (Default value = None)
    **kwargs :
        

    Returns
    -------

    """
    if ax is None:
        ax = plt.gca()
        
    lcolor = kwargs.get('color', 'C0')
    
    if add:
        ax.plot(run_sum, **kwargs)
        ax.vlines(es_idx, 0, run_sum[es_idx], color=lcolor, linestyle='--', lw=.8)
    else:
        ax.plot(run_sum, **kwargs)
        ax.axhline(y=0, linestyle='-', c='.15', lw=0.5)
        ax.vlines(es_idx, 0, run_sum[es_idx], color=lcolor, linestyle='--', lw=.8)
        ax.tick_params(labelsize='x-small')
        ax.set_ylabel('ES', fontsize='small')
    
    return ax
    
def _stats_legend(nes: float, 
                pval: float,
                leg_kw: dict = None):
    
    """Turn text for normalized enrichment score into a legend artist for easy placement in the axes.

    Parameters
    ----------
    nes: float :
        
    pval: float :
        
    leg_kw: dict :
         (Default value = None)

    Returns
    -------

    """
    
    leg_prop = {'loc':0, 
                'handlelength':0, 
                'handletextpad':0, 
                'labelspacing':0.2, 
                'edgecolor':'.15', 
                'fontsize':'xx-small', 
                'title_fontsize':'x-small',
                'labelcolor':'0.15',
                }
    
    if leg_kw is not None:
        leg_prop.update(leg_kw)
            
    ptch = mpl.patches.Patch(color='w')
    nes, pval = f"NES: {nes:1.2f}", f"p: {pval:1.1e}"
      
    leg = plt.legend([ptch]*2, [nes, pval], **leg_prop)
    leg.get_frame().set_linewidth(0.2)
    plt.setp(leg.get_title(), color=leg_prop.get('labelcolor'))
        
    return leg

def _add_reg_legend(color_pos:str, color_neg:str, 
                    leg_kw:dict=None):
    """

    Parameters
    ----------
    color_pos:str :
        
    color_neg:str :
        
    leg_kw:dict :
         (Default value = None)

    Returns
    -------

    """
    
    leg_prop = {'loc':1, #default position is top right
                'edgecolor':'.15', 
                'labelspacing':0.2,
                'handlelength':0.75, 
                'handletextpad':0.4,
                'fontsize':'xx-small', 
                'title':'Targets',
                'title_fontsize':'x-small'}
    
    if leg_kw is not None:
        leg_prop.update(leg_kw)
        
    handles = [Line2D([0],[0], lw=2, color=color_pos), Line2D([0],[0], lw=2, color=color_neg)]

    leg = plt.legend(handles, ['positive', 'negative'], **leg_prop)
    leg.get_frame().set_linewidth(0.2)
    
    return leg

    
def _plot_ledge_labels(xlims: tuple, 
                       genes: tuple, 
                       upper: bool = True,
                       highlight: tuple = None,
                       trim_ledge: int=None,
                       ax = None, 
                       **kwargs):
    """

    Parameters
    ----------
    xlims: tuple :
        
    genes: tuple :
        
    upper: bool :
         (Default value = True)
    highlight: tuple :
         (Default value = None)
    trim_ledge: int :
         (Default value = None)
    ax :
         (Default value = None)
    **kwargs :
        

    Returns
    -------

    """
    if ax is None:
        ax = plt.gca()
            
    ax.set_xlim(xlims)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    if trim_ledge is not None:
        assert len(genes)>=trim_ledge, 'Sorry, fewer genes in leading edge than you would like to limit'
        genes = genes[:trim_ledge]

    # Heuristic, spread out the genes along relative xaxis, add some jitter for yaxis position
    xpos = np.linspace(0.02, 0.98, len(genes))
    ypos = 0.5 + (np.random.normal(0, 0.05, len(genes)))
    
    if upper:
        ax.spines['bottom'].set_visible(False)
    else:
        ax.spines['top'].set_visible(False)
    
    if highlight is not None:
        assert isinstance(highlight, tuple), 'Need a tuple of strings, i.e. gene symbols'
        assert len(set(genes).intersection(set(highlight))) > 0, 'None of your genes found in leading edge'
        for x, y, g in zip(xpos, ypos, genes):
            if g in highlight:
                ax.annotate(text=g, xy=(x,y), xycoords='axes fraction', 
                            c='r', fontweight='bold', **kwargs)
            else: 
                 ax.annotate(text=g, xy=(x,y), xycoords='axes fraction', **kwargs)
                
    else:
        [ax.annotate(text=g, xy=(x,y), xycoords='axes fraction', **kwargs) for x, y, g in zip(xpos, ypos, genes)]
    
    return ax
       
 # Fix FDR for plotting
def _fdr_formatter(val: float):
    """

    Parameters
    ----------
    val: float :
        

    Returns
    -------

    """
    if val < 2e-16:
        return r'<2$e^{-16}$'
    else:
        a, b = f'{val:.1e}'.split('e')
        b = int(b)
    return fr'${a}e^{{{b}}}$'

def _prepare_multi_gseareg(linelengths:float, 
                                number_of_ys:int, 
                                space:float=0.2, 
                                start:float=0):
    """

    Parameters
    ----------
    linelengths:float :
        
    number_of_ys:int :
        
    space:float :
         (Default value = 0.2)
    start:float :
         (Default value = 0)

    Returns
    -------

    """
    
    lineoffsets = []
    tmp = start
    for i in range(number_of_ys*2):
        if i == 0:
            lineoffsets.append(start)
        elif i%2!=0:
            tmp += linelengths
            lineoffsets.append(tmp)
        else:
            tmp += space + linelengths
            lineoffsets.append(tmp)
        
    ytick_pos = [x for i, x in enumerate(lineoffsets) if i%2!=0]
    
    return lineoffsets, ytick_pos

def _format_xaxis_ges(ges_length:int, ax=None):
    """

    Parameters
    ----------
    ges_length:int :
        
    stepsize:int :
         (Default value = 1000)
    ax :
         (Default value = None)

    Returns
    -------

    """
    
    if ax is None:
        ax = plt.gca()
    
    if ges_length >= 10_000:
        stepsize = 1000
    else:
        stepsize = 500
    xticks = np.percentile(np.arange(0, ges_length, stepsize), [0, 50, 100])
    xtick_aln = ['left', 'center', 'right']
    
    ax.set_xticks(xticks)
    for i, tick in enumerate(ax.xaxis.get_majorticklabels()):
        tick.set_horizontalalignment(xtick_aln[i])  
    ax.xaxis.set_tick_params(labelsize='x-small')
    
    return ax




def _color_light_or_dark(rgba_in:np.ndarray)-> str:
    """[For plotting purposes, we determine whether a color is light or dark and adjust its text color accordingly.
    Also see https://stackoverflow.com/questions/22603510/is-this-possible-to-detect-a-colour-is-a-light-or-dark-colour]

    Args:
        rgba_in ([np.ndarray]): [A numpy array containing RGBA as returned by matplotlib colormaps]

    Returns:
        [str]: [A string: w for white or k for black]
    """
    r,g,b,_ = rgba_in*255
    hsp = np.sqrt(0.299 * (r * r) + 0.587 * (g * g) + 0.114 * (b * b))
    if (hsp>127.5):
        # light color, return black for text
        return 'k'
    else:
        # dark color, return white for text
        return 'w'



def _prepare_nes_colors(stats:pd.DataFrame,
                        norm_kw: dict=None, 
                        pcm_kw: dict=None):
    
    nes = stats['NES'].values
    
    pcm_prop = {'edgecolor':'.25', 'lw':.5}
    
    if all(nes>=0):
        norm_prop = {'vmin':0, 'vmax':np.max(nes)}
        pcm_prop.update({'cmap':'Reds'})
        
    elif all(nes<=0):
        norm_prop = {'vmin':np.min(nes), 'vmax':0}
        pcm_prop.update({'cmap':'Blues_r'})
        
    else:
        norm_prop = {'vmin':np.min(nes), 'vcenter':0, 'vmax':np.max(nes)}
        pcm_prop.update({'cmap':'PuOr_r'})
        
        
    if norm_kw is not None:
        norm_prop.update(norm_kw)
            
    if pcm_kw is not None:
        pcm_prop.update(pcm_kw)        
            
    if all(nes>=0) | all(nes<=0):
        norm = colors.Normalize(**norm_prop)
    else:
        norm = colors.TwoSlopeNorm(**norm_prop)
       
    norm_map = norm(nes)
    cmap = plt.cm.get_cmap(pcm_prop.get('cmap'))
    color_rgbas = cmap(norm_map)
        
        # Add color for NES text
    stats['color'] = [_color_light_or_dark(rgba) for rgba in color_rgbas]
    
    return norm, pcm_prop  
        

### Functions for leading edge plots
from matplotlib.transforms import (Bbox, TransformedBbox, blended_transform_factory)
from mpl_toolkits.axes_grid1.inset_locator import (BboxPatch, BboxConnector, BboxConnectorPatch)
import matplotlib as mpl

def _connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 line_kw=None, patch_kw=None):
    """

    Parameters
    ----------
    bbox1 :
        
    bbox2 :
        
    loc1a :
        
    loc2a :
        
    loc1b :
        
    loc2b :
        
    line_kw :
         (Default value = None)
    patch_kw :
         (Default value = None)

    Returns
    -------

    """
    
    # get a default for the lines
    line_prop = {"clip_on": False, 'color':'k', 'lw':0.5}
    if line_kw is not None:
        line_prop.update(line_kw)
    
    patch_prop = {"clip_on": False, 'color':'C0', 'alpha':0.1, 'ec':'none'}
    if patch_kw is not None:
        patch_prop.update(patch_kw)


    c1 = BboxConnector(
        bbox1, bbox2, loc1=loc1a, loc2=loc2a, **line_prop)
    c2 = BboxConnector(
        bbox1, bbox2, loc1=loc1b, loc2=loc2b, **line_prop)

    bbox_patch1 = BboxPatch(bbox1, **patch_prop)
    bbox_patch2 = BboxPatch(bbox2, **patch_prop)

    # This should not be modified
    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b, 
                           clip_on=False, ec='none')

    return c1, c2, bbox_patch1, bbox_patch2, p


def zoom_effect(ax1:mpl.axes.Axes, 
                ax2:mpl.axes.Axes, 
                upper: bool=True, 
                line_kw:dict=None, 
                patch_kw:dict=None):
    """ax1 : the main axes
    ax2 : the zoomed axes
    
    The xmin & xmax will be taken from the
    ax1.viewLim.

    Parameters
    ----------
    ax1:mpl.axes.Axes :
        
    ax2:mpl.axes.Axes :
        
    upper: bool :
         (Default value = True)
    line_kw:dict :
         (Default value = None)
    patch_kw:dict :
         (Default value = None)

    Returns
    -------

    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(ax2.transData, tt)
    
    # with this option, we can switch connections modes
    if upper:
        loc1a, loc2a, loc1b, loc2b = (3, 2, 4, 1)
    else:
        loc1a, loc2a, loc1b, loc2b = (1, 4, 2, 3)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    c1, c2, bbox_patch1, bbox_patch2, p = _connect_bbox(
        mybbox1, mybbox2,
        loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
        line_kw=line_kw, patch_kw=patch_kw)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p