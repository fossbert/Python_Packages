import numpy as np
import pandas as pd
from scipy.stats import norm


def aREA(dset, regulon, minsize=20):

    """This function takes an expression matrix or signature, respectively, and
    computes analytic rank-based enrichment analysis as described in Alvarez et al., Nat
    Genetics, 2016"""

    # transform dset to rank matrices
    t1, t2 = get_tmats(dset)

    # get weight matrices from MoR and Likelihood
    mor, lik, lik_scaled = get_mor_lik(regulon, minsize=minsize)

    # calculate three-tailed enrichment score
    s1 = calc_2TES(t2, mor, lik_scaled) #2Tailed ES
    s2 = calc_1TES(t1, mor, lik_scaled) #1tailed ES
    s3 = calc_3TES(s1, s2)

    # Normalize
    nes = calc_nes(s3, lik)

    return nes


def calc_nes(s3, lik):
    """This function normalized the Three-tailed
    enrichment score using the weights from the interaction confidence"""

    # likelihood weights
    lwt = np.sqrt(np.sum(lik.values**2,axis=0))[:,np.newaxis]

    nes = s3 * lwt

    return nes



def calc_3TES(s1, s2):

    """This function calculates the Three-Tailed enrichment score from two ES DFs.
    It will expect the Two-Tail ES as s1 and One-Tail ES as s2"""

    # Extract the signs of the Two-tail enrichment scores
    s1_signs = s1.copy()
    s1_signs[s1_signs>=0] = 1
    s1_signs[s1_signs<0] = -1

    # Create final ES
    s3 = (s1.abs() + s2 * (s2 > 0)) * s1_signs
    return s3



def calc_1TES(t1, mor, lik_scaled):

    """This function calculates the one-tail enrichment score DataFrame"""
    # Weight matrix irrespective of direction
    wm1 = (1 - mor.abs()) * lik_scaled

    # Align matrices
    pos = t1.index.intersection(wm1.index)
    t1 = t1.loc[pos]
    wm1 = wm1.loc[pos]

    # Multiply
    s2 = wm1.T @ t1
    return s2


def calc_2TES(t2, mor, lik_scaled):

    """This function calculates the two-tail enrichment scores"""

    # Create two-tailed weighted MoR DF
    wm2 = mor * lik_scaled

    # Align DFs
    pos = t2.index.intersection(wm2.index)
    t2 = t2.loc[pos]
    wm2 = wm2.loc[pos]

    # Multiply
    s1 = wm2.T @ t2
    return s1



def get_mor_lik(regulon: dict, minsize: int = 20):

    """This function extracts numeric weight DataFrames from a regulon dictionary for further matrix multiplication steps"""

    mors = []
    liks = []

    assert isinstance(regulon, dict), 'Need a dictionary containing DataFrames as regulon input'

    for key, val in regulon.items():

        # implement filter step here
        if len(val) >= minsize:
            mors.append(pd.Series(data=val['mor'].values,
                    index=val['target'].values,
                    name=key))
            liks.append(pd.Series(data=val['likelihood'].values/np.max(val['likelihood']),
                    index=val['target'].values,
                    name=key))

    # join using outer join to keep all targets
    mor_df = pd.concat(mors, axis=1)
    mor_df[mor_df.isnull()] = 0

    lik_df = pd.concat(liks, axis=1)
    lik_df[lik_df.isnull()] = 0

    # Scale likelihood DF for column sums
    lik_scaled = lik_df / np.sum(lik_df.values, axis=0, keepdims=True)

    return mor_df, lik_df, lik_scaled

def get_tmats(dset):

    """Transforms an expression matrix or gene expression signature"""

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

    elif isinstance(dset, pd.DataFrame):
        t2q = pd.DataFrame(norm.ppf(t2.values),
                  index=t2.index,
                  columns=t2.columns)

        t1q = pd.DataFrame(norm.ppf(t1.values),
                  index=t1.index,
                  columns=t1.columns)

    return t1q, t2q




def genesets2regulon(genesets: dict, 
                     minsize: int = 20, 
                     maxsize: int = None):

    """This function generates a regulon dictionary suitable for aREA from a 'regulary' dictionary
    of pathway names as keys and a list of the related gene symbols as values."""

    assert isinstance(genesets, dict), 'A dictionary is needed here!'
    assert isinstance(minsize, int), 'minsize needs to be an integer value!'

    reg = {}

    for key, val in genesets.items():
        ns = len(val)
        if maxsize != None:
            assert isinstance(maxsize, int), 'maxsize needs to be an integer value!'
            if (ns >= minsize) & (ns <= maxsize):
                mor = np.ones(ns)
                reg[key] = pd.DataFrame(data=zip(val, mor, mor/ns),
                                        columns=['target', 'mor', 'likelihood'])

        else:
            if ns >= minsize:
                mor = np.ones(ns)
                reg[key] = pd.DataFrame(data=zip(val, mor, mor/ns),
                                    columns=['target', 'mor', 'likelihood'])

    return reg
