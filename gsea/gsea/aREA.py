import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Union, Tuple


def aREA(dset: Union[pd.DataFrame, pd.Series], 
         regulon: pd.DataFrame, 
         minsize: int = 20)->pd.DataFrame:
    
    """[This function takes an , respectively, and
    computes analytic rank-based enrichment analysis as described in Alvarez et al., Nat
    Genetics, 2016]

    Args:
      dset: Union[pd.DataFrame:pd.Series]: Numeric expression matrix or single signature.
      regulon: dict: A regulon, i.e. dictionary containing DataFrames for each regulator or pathway which holds
    the information on targets, mode of regulation and likelihoods.
      minsize: int: Integer specifying the minimum required number of target genes for a regulator be to considered for
    enrichment analysis. (Default value = 20)

    Returns:
      pd.Series:pd.DataFrame: A pandas Series or DataFrame of Normalized Enrichment Scores.

    """
    
    # transform dset to rank matrices
    t1, t2 = get_tmats(dset)

    # get weight matrices from MoR and Likelihood
    mor, wts, wtss = get_mor_wts(regulon, minsize=minsize)

    # calculate three-tailed enrichment score
    s1 = calc_2TES(t2, mor, wtss) #2Tailed ES
    s2 = calc_1TES(t1, mor, wtss) #1tailed ES
    s3 = calc_3TES(s1, s2)

    # Normalize
    nes = calc_nes(s3, wts)

    return nes


def calc_nes(s3:pd.DataFrame, 
             lik:pd.DataFrame)->pd.DataFrame:
    """
        This function normalizes the three-tailed enrichment score using the weights from the interaction confidence

    Args:
    s3:pd.DataFrame: Numeric DataFrame containing three-tailed enrichment scores.
    lik:pd.DataFrame: Numeric DataFrame containing likelihoods between regulators and targets.

    Returns:
      pd.DataFrame: A DataFrame of Normalized Enrichment Scores

    """

    # likelihood weights
    lwt = np.sqrt(np.sum(lik.values**2,axis=0))[:,np.newaxis]

    nes = s3 * lwt

    return nes


def calc_3TES(s1:pd.DataFrame, 
              s2:pd.DataFrame)->pd.DataFrame:

    """This function calculates the three-tailed enrichment score from two enrichment score (ES) DataFrames.
        It will expect the two-tail ES as s1 and one-tail ES as s2.

    Args:
      s1:pd.DataFrame: Numeric DataFrame of two-tailed enrichment scores
      s2:pd.DataFrame: Numeric DataFrame of one-tailed enrichment scores

    Returns:
      pd.DataFrame: Numeric DataFrame containing three-tailed enrichment scores

    """

    # Extract the signs of the two-tail enrichment scores
    s1_signs = s1.copy()
    s1_signs[s1_signs>=0] = 1
    s1_signs[s1_signs<0] = -1

    # Create final ES
    s3 = (s1.abs() + s2 * (s2 > 0)) * s1_signs
    return s3



def calc_1TES(t1:pd.DataFrame, 
              mor:pd.DataFrame, 
              lik_scaled:pd.DataFrame)->pd.DataFrame:

    """[This function calculates the one-tail enrichment score DataFrame]

    Args:
      t1:pd.DataFrame:Numeric DataFrame containing T1 ranked data, i.e. scaled expression values which are highest
    when the input approaches the negative and positive extremes, respectively.
      mor:pd.DataFrame: Numeric DataFrame containing the Mode of Regulation information between regulators and their targets
      lik_scaled:pd.DataFrame: Numeric DataFrame containing scaled weights, i.e. likelihoods bwetween each regulator and target.

    Returns:
      pd.DataFrame: A numeric DataFrame containing one-tailed enrichment score results

    """
    # Weight DataFrame irrespective of direction
    wm1 = (1 - mor.abs()) * lik_scaled
    # Align matrices
    pos = t1.index.intersection(wm1.index)
    t1 = t1.loc[pos]
    wm1 = wm1.loc[pos]

    # Multiply
    s2 = wm1.T @ t1
    return s2


def calc_2TES(t2:pd.DataFrame, 
              mor: pd.DataFrame, 
              lik_scaled: pd.DataFrame)->pd.DataFrame:

    
    """This function calculates the two-tail enrichment score DataFrame

    Args:
      t2:pd.DataFrame: Numeric DataFrame containing T2 ranked data, i.e. scaled expression values from 0 (low) to 1 (high).
      mor: pd.DataFrame: Numeric DataFrame containing the Mode of Regulation information between regulators and their targets.
      lik_scaled: pd.DataFrame: Numeric DataFrame containing scaled weights, i.e. likelihoods bwetween each regulator and target.

    Returns:
      pd.DataFrame: A numeric DataFrame containing two-tailed enrichment score results.

    """
    # Create two-tailed weighted MoR DF
    wm2 = mor * lik_scaled

    # Align DFs
    pos = t2.index.intersection(wm2.index)
    t2 = t2.loc[pos]
    wm2 = wm2.loc[pos]

    # Multiply
    s1 = wm2.T @ t2
    return s1


def get_mor_wts(regulon: pd.DataFrame, 
                minsize: int)-> Tuple[pd.DataFrame, pd.DataFrame]:
 
    if not isinstance(regulon, pd.DataFrame):
      raise ValueError('Need a pandas DataFrame as a regulon')
    if len(regulon.columns)!=4:
      raise ValueError('Regulon DataFrame needs 4 columns: source, target, mor, likelihood')
    
    if not all(regulon.columns.values == np.array(['source', 'target', 'mor', 'likelihood'])):
      regulon.columns = ['source', 'target', 'mor', 'likelihood']

    # Wrangling
    df = regulon.copy()
    df.set_index(['source', 'target'], inplace=True)
    mor = df['mor'].unstack(level=0)
    wts = df['likelihood'].unstack(level=0)
   
    # Implement filter step
    keep = mor.notnull().sum() >= minsize
    mor = mor.loc[:,keep]
    mor.fillna(0, inplace=True)
    wts = wts.loc[:,keep]
    wts.fillna(0, inplace=True)  
    # Scale likelihood values
    wts = wts / np.max(wts.values, axis=0, keepdims=True)

    # Scale likelihood DF for column sums
    wtss = wts / np.sum(wts.values, axis=0, keepdims=True)

    return mor, wts, wtss


def get_tmats(dset: Union[pd.DataFrame, pd.Series])-> Tuple[pd.DataFrame, pd.DataFrame]:

    """From a single gene expression Series or gene expression DataFrame, this function calculates
    T1 and T2 numeric DataFrames, i.e. the scaled expression rank of every gene within the sample. The ranked values are
    transformed to quantiles from the normal distribution.

    Args:
      dset: Union[pd.DataFrame:pd.Series]: Numeric Series or DataFrame with genes as Index.

    Returns:
      Tuple: Tuple containing two DataFrames: T1 and T2 ranks

    """
    assert isinstance(dset, pd.Series) or isinstance(dset, pd.DataFrame), 'Wrong data input'

    # rank computation
    t2 = dset.rank(method='first') / (len(dset)+1)
    t1 = np.abs(t2 - 0.5) * 2
    t1 = t1 + (1 - np.max(t1))/2

    # transform to quantiles of a normal distribution
    if isinstance(dset, pd.Series):
        t2q = pd.DataFrame(norm.ppf(t2.values),
                  index=t2.index,
                  columns=[t2.name])

        t1q = pd.DataFrame(norm.ppf(t1.values),
                  index=t1.index,
                  columns=[t1.name])

    # No further checking necessary, this is checked during Class instantiation
    else:
        t2q = pd.DataFrame(norm.ppf(t2.values),
                  index=t2.index,
                  columns=t2.columns)

        t1q = pd.DataFrame(norm.ppf(t1.values),
                  index=t1.index,
                  columns=t1.columns)

    return t1q, t2q
