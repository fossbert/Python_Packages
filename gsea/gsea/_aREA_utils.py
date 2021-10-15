import numpy as np
import pandas as pd
from typing import Union

"""Helper functions used throughout the package"""

def gene_sets_to_regulon(genesets: dict, 
                         minsize: int = 20)-> pd.DataFrame:

    """This function generates a regulon dictionary suitable for aREA from a 'regulary' dictionary
    of pathway names as keys and a list of the related gene symbols as values.

    Args:
      genesets: dict: Dictionary of pathway names (=keys) and lists of pathway member genes (=values).
      minsize: int: Integer specifying the minimum number of genes or targets a pathway or regulator should have (Default value = 20).
      maxsize: int: Integer specifying the maximum number of genes or targets a pathway or regulator should have (Default value = None).

    Returns:
        dictionary: A dictionary of DataFrames with information on pathway/regulator targets, mode of regulation (always=1) and likelihoods (always=1/N targets).

    """

    reg = {}

    for key, val in genesets.items():
      # duplicate values will mess things up ! 
        set_in = set(val)
        ns = len(val)
        
        if ns>= minsize:
          reg[key] = _gene_set_to_df(set_in, ns)
     
    df = pd.concat(reg)
    df.index.set_names('source', level=0, inplace=True)
    df.reset_index(level=0, inplace=True)
    
    return df
  
def _gene_set_to_df(gene_set: set, ns: int):
  
  mor = np.ones(ns)
    
  return pd.DataFrame(data=zip(gene_set, mor, mor/ns),
                      columns=['target', 'mor', 'likelihood'])



def _prep_ges(ges: Union[pd.Series,pd.DataFrame], 
              asc_sort: bool=True) -> pd.Series:

    """[This function is used to clean up signatures containing NA's in their index or merging
    scores values for duplicated genes by averaging them.]

    Args:
      ges: pd.Series or pd.DataFrame: 
      asc_sort: bool:  (Default value = True)

    Returns:
      pd.Series|pd.DataFrame: cleaned gene expression signature(s)
    """
    
    assert (isinstance(ges, pd.Series) or isinstance(ges, pd.DataFrame)), 'Need pandas Series or DataFrame!'
        
    if isinstance(ges, pd.Series):
      
      if ges.name is None:
        ges.name = 'ges'
      if ges.isnull().any():
        ges.dropna(inplace=True)    
        
      if ges.index.duplicated().any():
        ges = ges.groupby(level=0).mean() # collapse by averaging
      
      ges.sort_values(ascending=asc_sort, inplace=True)
      
    else:
      if ges.isnull().any().any():
        ges.dropna(inplace=True)
          
      if ges.index.duplicated().any(): 
        ges = ges.groupby(level=0).mean()
                 
    return ges
    