
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib.cm import coolwarm, ScalarMappable
import matplotlib.colors as mcolors
from matplotlib import ticker
from aREA import aREA, genesets2regulon
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

"""
This module implements gene set enrichment functionality for one-tailed gene sets. 
"""

class Gsea1T:
    
    """"""

    def __init__(self, 
                 ges: pd.Series, 
                 gene_set: list, 
                 ascending: bool = False, 
                 weight: float = 1):

        self.asc_sort = ascending
        self.weight = weight

        if not isinstance(ges, pd.Series):
            raise TypeError('Need an indexed pandas Series, please.')
        else:
            self.ges = _prep_ges(ges, self.asc_sort)
            self.ns = len(self.ges)
            self.along_scores = [*range(self.ns)]

        if not np.in1d(gene_set, ges.index).any():
            raise ValueError('None of the genes in gene set found in GES index')
        else:
            self.gs_org = gene_set
            self.gs_final = [g for g in gene_set if g in self.ges.index]

        self.gs_idx = self._find_hits()

        self.rs = self._derive_rs()
        self.es_idx = np.abs(self.rs).argmax()

        self.gs_reg = genesets2regulon({'GS':self.gs_org})
        self.aREA_nes = aREA(self.ges,
                            self.gs_reg).iloc[0][0]

        self.pval = norm.sf(np.abs(self.aREA_nes))*2
        self.ledge = self._get_ledge()


    def __repr__(self):

        """String representation for the Gsea1T class"""

        return """Gsea1T(GES length: {ngenes}, Gene set:
        {gslength}, Overlap: {ngenesinges})""".format(ngenes = self.ns,
                   gslength = len(self.gs_org),
                   ngenesinges = len(self.gs_final))

    def _find_hits(self):

        """Finds the positions of a gene set in a given gene expression signaure list"""

        return {gene: idx for gene, idx in zip(self.ges.index, self.along_scores) if gene in self.gs_final}


    def _derive_rs(self):

        """Derives the running sum for plotting"""

        idx = list(self.gs_idx.values())

        Nr = np.sum(np.abs((self.ges.values[idx]**self.weight))) # normalization factor
        Nh = self.ns - len(self.gs_final) # non-hits
        tmp = np.repeat(-1/Nh, self.ns)
        tmp[idx] = np.abs(self.ges.values[idx]**self.weight)/Nr
        rs = np.cumsum(tmp) # running sum

        return rs

    def _get_ledge(self):

        if self.aREA_nes > 0:
            return {gene: idx for gene, idx in self.gs_idx.items() if idx <= self.es_idx}
        else:
            return {gene: idx for gene, idx in self.gs_idx.items() if idx >= self.es_idx}


    def plot(self,
             figsize: tuple=(3,3),
             bar_alpha: float=0.7,
             phenotypes:tuple = ('A', 'B'),
             colors: tuple = ('.75', '#439D75')
             ):

        """This will return a figure object containing 3 axes displaying the gene expression signature,
        the gene set indices and the running sum line"""

        fig = plt.figure(figsize=figsize, tight_layout=True)

        gs = gridspec.GridSpec(3, 1, 
                               height_ratios=[2, 1, 5], 
                               hspace=0, 
                               figure=fig)

        # first graph
        ax1 = fig.add_subplot(gs[0])
        ax1.fill_between(self.along_scores, self.ges.values, color=colors[0])
        ax1.set_xticks([])
        ax1.set_ylabel('Gene score', fontsize='small')
        ax1.axhline(y=0, linestyle=':', c='.5')
        p1, p2 = phenotypes
        ax1.annotate(p1, xy=(0.05, 0.05), xycoords='axes fraction', ha='left', va='bottom')
        ax1.annotate(p2, xy=(0.95, 0.95), xycoords='axes fraction', ha='right', va='top')

        # second graph: bars to indicate positions of individual genes
        ax2 = fig.add_subplot(gs[1])
        ax2.eventplot(list(self.gs_idx.values()), color=colors[1], alpha=bar_alpha)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[2])

        # Set up a legend for NES and p-value
        ptch = mpl.patches.Patch(color='w')
        nes, pval = "NES: {:1.2f}".format(self.aREA_nes), "p: {:1.1e}".format(self.pval)

        ax3.plot(self.rs, color=colors[1])
        ax3.axhline(y=0, linestyle='-', c='.5', lw=0.8)
        ax3.vlines(self.es_idx, 0, self.rs[self.es_idx], color=colors[1], linestyle='--', lw=.8)
        ax3.set_ylabel('ES')
        ax3.legend([ptch, ptch], [nes, pval],
        handlelength=0, handletextpad=0, loc=0, fontsize='x-small')
        ax3.xaxis.set_major_locator(ticker.FixedLocator(np.linspace(0, self.ns, 4)))
        ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:1.1f}K'.format(x*1e-3)))
        ax3.xaxis.set_tick_params(labelsize='x-small')

        return fig


class Gsea1TMultSig:
     
    def __init__(self, 
                 dset:pd.DataFrame, 
                 gene_set:list,  
                 ordered: bool=True):
                
        if not isinstance(dset, pd.DataFrame):
            raise TypeError('Need an indexed pandas DataFrame, please.')
            
        else: 
            self.dset = self._prep_dset(dset)
            self.ns = len(self.dset)
            self.samples = self.dset.columns.values
        
        if not np.in1d(gene_set, dset.index).any():
            raise ValueError('None of the genes in gene set found in dset DataFrame index')
        else:
            self.gs_org = gene_set
            self.gs_final = [g for g in gene_set if g in self.dset.index]
        
        self.gs_idx = self._find_hits()
        
        self.gs_reg = genesets2regulon({'GS':self.gs_org})
        
        self.stats = self._get_stats()
        
        if ordered:
            self.stats.sort_values('NES', inplace=True)
            self.stats.reset_index(inplace=True)
            
    
    def __repr__(self):
        
        """String representation for the Gsea1TMult class"""

        return """Gsea1TMult(Number of genes: {ngenes}, Number of signatures: {nsigs}, Gene set:
        {gslength}, Overlap: {ngenesindset})""".format(ngenes = self.ns, 
                                                      nsigs = len(self.samples),
                                                       gslength = len(self.gs_org),
                                                       ngenesindset = len(self.gs_final))
    
    
    def _prep_dset(self, dset):
        
        """This function removes duplicate indices and NAs, 
        then sorts according to the specified sorting direction"""
        
        if dset.isnull().any().any():
            dset.dropna(inplace=True)
            
        if dset.index.duplicated().any():
            dset = dset.reset_index().groupby('index').mean() # collapse by averaging
            
        return dset
    
    def _find_hits(self):
        
        """Finds the positions of a gene set in a given gene expression signaure matrix"""
        
        # rank order all signatures 
        hits = self.dset.apply(lambda x: x.sort_values().argsort(), axis=0)
        
        # subset to those in the gene set
        hits = hits.loc[self.gs_final]
        
        # Return the indices for the gene set in each signature as a 
        # numpy array of arrays, transverse for plotting
        return hits.values.T
    
    def _get_stats(self):
        
        """Computes normalized enrichment scores and some tools for later visualization"""
        
        nes = aREA(self.dset, self.gs_reg).iloc[0].values.T.ravel() # get a flattened one-dimensional array
        pvals = norm.sf(np.abs(nes))*2 # retrieve two-sided p-value from normal distribution
        fdrs = multipletests(pvals, method = 'fdr_bh')[1] #FDR
           
        return pd.DataFrame(data=zip(nes, pvals, fdrs, self.gs_idx, self.samples),
                            columns=['NES', 'pval', 'fdr', 'positions', 'signature_name'])
        
        
    def plot(self, 
             bar_alpha: float = 0.5,
             figsize: tuple = (3.5, 3),
             norm_kws: dict = None):
         
        """This will return a figure object containing 4 axes displaying the gene expression signature,
        the gene set indices and the running sum line"""
        
        
        df = self.stats.copy()
        
        # In this first bit, we determine the NES color scheme
        if norm_kws is None:
            norm_kws = {'vcenter':0, 
                        'vmin':np.min(df['NES'].values), 
                        'vmax':np.max(df['NES'].values)}
            
        norm = mcolors.TwoSlopeNorm(**norm_kws)
        norm_map = norm(df['NES'].values)
        
        df['color'] = pd.cut(norm_map, 
                             bins=[0, 0.1, 0.9, 1], 
                             ordered=False,
                             labels=['w','k','w'], 
                             include_lowest=True)
        
        fig = plt.figure(figsize=figsize) 
        
        gs = fig.add_gridspec(nrows=2, 
                      ncols=2, 
                      hspace=0, 
                      wspace=0.05,
                      width_ratios=[10,1],
                      height_ratios=[1,40])
        
                # Plot 1: Illustrate the ordered signature as a colorbar
        ax1 = fig.add_subplot(gs[0,0])
        cb = fig.colorbar(ScalarMappable(cmap=coolwarm), 
             orientation='horizontal', ticks=[], 
             cax=ax1)
        cb.outline.set_visible(False)
        ax1.text(0, 0.5, 'Low', color='w', ha='left', va='center', fontsize='x-small')
        ax1.text(1, 0.5, 'High', color='w', ha='right', va='center', fontsize='x-small')
    
        # Plot 2: Illustrate targets
        ax2 = fig.add_subplot(gs[1,0])
        ax2.eventplot(df['positions'].values, linelengths=3/4, color='.25', alpha=bar_alpha)
        ax2.set_yticks(range(len(df)))
        ax2.set_ylim([-0.5, len(df)-0.5])
        ax2.set_yticklabels(df['signature_name'].values, fontsize='x-small')
        ax2.set_xlim([-.5, self.ns+.5])
        # x-axis formating
        ax2.xaxis.set_major_locator(ticker.FixedLocator(np.linspace(0, self.ns, 4)))
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:1.1f}K'.format(x*1e-3)))
        ax2.xaxis.set_tick_params(labelsize='x-small')
        
        # remove spines
        for spine in ['right', 'left', 'top']:
            ax2.spines[spine].set_visible(False)

        # Plot 3: NES heatmap

        ax3 = fig.add_subplot(gs[1,1])
        ax3.pcolormesh(df['NES'].values[:,np.newaxis], 
               norm=norm, 
               cmap='PuOr_r', 
               edgecolor='.25', lw=.5)

        for row in df.itertuples():
            ax3.text(0.5, row.Index + 0.5, '{:1.2f}'.format(row.NES),
                fontsize='xx-small', c=row.color,
                ha='center', va='center')
            ax3.text(1, row.Index + 0.5, '{:1.1e}'.format(row.fdr),
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
    
    """"""

    def __init__(self, 
                 ges: pd.Series, 
                 gene_sets: dict, 
                 ordered: bool = True,
                 strip_gs_names: bool = True,
                 minsize:int = 20):

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
        
        self.gs_reg = genesets2regulon(self.gene_sets, minsize=self.minsize)
              
        self.gs_idx = self._find_hits(self.ges, self.gene_sets)
        
        self.stats = self._get_stats(self.ges, self.gs_reg)
        
        if strip_gs_names:
            self.stats['gene_set'] = self.stats['gene_set'].str.replace('^[A-Z]+_', '')
        
        if ordered:
            self.stats.sort_values('NES', inplace=True)
            self.stats.reset_index(inplace=True)
            

    def __repr__(self):

        """String representation for the Gsea1TMultSet class"""

        return """Gsea1TMultSet(GES length: {ngenes}, Gene sets in:
        {gene_sets_in}, Gene sets used: {gene_sets_used})""".format(ngenes = self.ns,
                   gene_sets_in = self.n_org_gene_sets,
                   gene_sets_used = len(self.gene_sets))
        
    def _prep_gene_sets(self, 
                        gene_sets: dict,
                        genes: pd.Index,
                        minsize: int) -> dict:
        
        """Go through the provided dict of gene sets and check whether at least minsize 
        genes of said gene set occur in the gene expression signature"""
        return {key:val for key, val in gene_sets.items() if np.in1d(val, genes).sum() >= minsize}

    def _find_hits(self, 
                   ges: pd.Series, 
                   gene_sets: dict):


        # This will have to be sorted first, which is the case in this Class
        ges = ges.argsort()
        
        return [ges[ges.index.intersection(val)].values for val in gene_sets.values()]
            
        
    def _get_stats(self, 
                   ges: pd.Series,
                   regulon: dict):
        
        """Computes normalized enrichment scores and some tools for later visualization"""
        
        nes = aREA(self.ges, self.gs_reg).values.ravel() # get a flattened one-dimensional array
        pvals = norm.sf(np.abs(nes))*2 # retrieve two-sided p-value from normal distribution
        fdrs = multipletests(pvals, method = 'fdr_bh')[1] #FDR
        positions = np.argsort(nes) # for ordering if needed
                   
        return pd.DataFrame(data=zip(nes, pvals, fdrs, positions, self.gs_idx, self.gs_names),
                            columns=['NES', 'pval', 'fdr', 'order', 'positions', 'gene_set'])
        
                    
    def plot(self, 
             phenotypes: tuple = ('A', 'B'),
             bar_alpha: float = 0.5,
             figsize: tuple = (3.5, 3),
             norm_kws: dict = None,
             ges_type:str = None):
         
        """This will return a figure object containing 4 axes displaying the gene expression signature,
        the gene set indices and the Normalized Enrichment scores and associated FDR"""
        
        df = self.stats.copy()
        
        if norm_kws is None:
            norm_kws = {'vcenter':0, 
                        'vmin':np.min(df['NES'].values), 
                        'vmax':np.max(df['NES'].values)}
        
        norm = mcolors.TwoSlopeNorm(**norm_kws)
        norm_map = norm(df['NES'].values)
        
        df['color'] = pd.cut(norm_map, 
                             bins=[0, 0.1, 0.9, 1], 
                             ordered=False,
                             labels=['w','k','w'], 
                             include_lowest=True)
        
        
        fig = plt.figure(figsize=figsize) 
        
        gs = fig.add_gridspec(nrows=2, 
                      ncols=2, 
                      hspace=0, 
                      wspace=0.05,
                      width_ratios=[10,1],
                      height_ratios=[1,8])
        
                # Plot 1: Illustrate the ordered signature as a colorbar
        ax1 = fig.add_subplot(gs[0,0])
        ax1.fill_between(self.along_scores, self.ges.values, color='.75')
        ax1.set_xticks([])
        if ges_type is None:
            ges_type = 'Gene score'
        ax1.set_ylabel(ges_type, fontsize='x-small')
        ax1.axhline(y=0, linestyle=':', c='.5')
        p1, p2 = phenotypes
        ax1.annotate(p1, xy=(0.05, 0.95), xycoords='axes fraction', ha='right', va='top')
        ax1.annotate(p2, xy=(0.95, 0.05), xycoords='axes fraction', ha='left', va='bottom')

        
        # Plot 2: Illustrate targets
        ax2 = fig.add_subplot(gs[1,0])
        ax2.eventplot(df['positions'].values, linelengths=3/4, color='.25', alpha=bar_alpha)
        ax2.set_yticks(range(len(df)))
        ax2.set_ylim([-0.5, len(df)-0.5])
        ax2.set_yticklabels(df['gene_set'].values, fontsize='x-small')
        ax2.set_xlim([-.5, self.ns+.5])
        # x-axis formating
        ax2.xaxis.set_major_locator(ticker.FixedLocator(np.linspace(0, self.ns, 4)))
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:1.1f}K'.format(x*1e-3)))
        ax2.xaxis.set_tick_params(labelsize='x-small')
        
        # remove spines
        for spine in ['right', 'left', 'top']:
            ax2.spines[spine].set_visible(False)

        # Plot 3: NES heatmap
            
        ax3 = fig.add_subplot(gs[1,1])
        ax3.pcolormesh(df['NES'].values[:,np.newaxis], 
               norm=norm,
               cmap='PuOr_r', 
               edgecolor='.25', lw=.5)

        for row in df.itertuples():
            ax3.text(0.5, row.Index + 0.5, '{:1.2f}'.format(row.NES),
                fontsize='xx-small', c=row.color,
                ha='center', va='center')
            ax3.text(1, row.Index + 0.5, '{:1.1e}'.format(row.fdr),
                fontsize='xx-small', c='k',
                ha='left', va='center')
            ax3.axis('off')

        # Plot 4: Not sure if necessary to open new plot, but use grid fully. Just titles for annotation
        ax4 = fig.add_subplot(gs[0,1])
        ax4.text(0.5, 0, 'NES', fontsize='x-small', c='k', ha='center', va='bottom')
        ax4.text(1.1, 0, 'FDR', fontsize='x-small', c='k', ha='left', va='bottom')
        ax4.axis('off')
        
        return fig

# TODO: This will probably go to _utils.py eventually

def _prep_ges(ges: pd.Series, asc_sort: bool=True) -> pd.Series:

    """[This function is used to clean up signatures containing NA's in their index or merging
    scores values for duplicated genes by averaging them.]

    Returns:
        [pd.Series]: [cleaned gene expression signature]
    """
    if ges.isnull().any():
        ges.dropna(inplace=True)

    if ges.index.duplicated().any():
        tmp = ges.reset_index(name='ges').groupby('index').mean() # collapse by averaging
        ges = pd.Series(tmp.ges.values, index=tmp.index)

    ges.sort_values(ascending=asc_sort, inplace=True)

    return ges


if __name__ == "__main__":
    ngenes = 10000
    
    ### Test for Gsea1T
    
    # genes = ['Gene' + str(i) for i in range(ngenes)]
    # np.random.shuffle(genes)
    # ges = pd.Series(np.random.normal(size=ngenes),
    #             index=genes, name='example')
    
    # gene_set = np.random.choice(ges.index, replace=False, size=50)
    
    # gsobj = Gsea1T(ges, gene_set)
    # # print(gsobj)
    # fig = gsobj.plot()
    # fig.savefig('Test-Gsea1T.pdf')
    
     ### Test for Gsea1TMultSig
    
    data = np.random.normal(size=(ngenes, 20))
    genes = ['Gene' + str(i) for i in range(ngenes)]
    samples = ['Sample' + str(i) for i in range(20)]
    np.random.shuffle(genes)
    dset = pd.DataFrame(data, 
                 index=genes,
                   columns=samples)

    gene_set = np.random.choice(dset.index, replace=False, size=50)
    
    gsobj = Gsea1TMultSig(dset, gene_set=gene_set)
    #print(gsobj.stats['positions'].values)
    fig = gsobj.plot(figsize=(2.5, 3.5), norm_kws={'vcenter':0, 'vmin':-5, 'vmax':5})
    fig.savefig('Test-Gsea1TMultSig.pdf')
           
    # Test for Gsea1TMultSet
    # dset = pd.read_table('../Analyses/AG_Reichert/Data/TUMOrganoidsCore_RNASeq/TUMOrganoidsCore_RawCounts_EsetCollapsedTR27571x27.tsv', index_col=0)
    # dset = (dset - dset.values.mean(1, keepdims=True)) / dset.values.std(1, keepdims=True)
    
    # ges = dset.loc[:,'B113.NA'].copy()
        
    # gene_sets = {}
    # for i in range(10):
    #     genes = np.random.choice(ges.index, replace=False, 
    #                              size=np.random.randint(50, 101, size=1))
    #     gene_sets['Set' + str(i)] = genes
    
    # gene_sets = {}

    # with open('../Data/HALLMARK_related/h.all.v7.4.symbols.gmt') as f:
    #     for line in f:
    #         tmp = line.rstrip('\n').split('\t')
    #         pw = tmp.pop(0)
    #         gene_sets[pw] = tmp[1:]
         
    # gsobj = Gsea1TMultSets(ges, gene_sets, minsize=75)
    
    # # print(gsobj.stats)
    
    # fig = gsobj.plot(figsize=(3, 6))
    # fig.savefig('Test-Gsea1TMultSet.pdf')
    