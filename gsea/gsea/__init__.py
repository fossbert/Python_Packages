"""
The gsea package provides classes and functions for gene set enrichment analysis.
"""

from .Gsea1T import Gsea1T, Gsea1TMultSets, Gsea1TMultSigs
from .Gsea2T import Gsea2T, GseaReg, GseaRegTMultSigs, GseaMultReg
from .aREA import aREA
from ._aREA_utils import genesets2regulon