# computing
from collections import namedtuple
from typing import Any, Sequence
import numpy as np
import pandas as pd

# plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from adjustText import adjust_text

# stats
from scipy.spatial import distance
from scipy.cluster import hierarchy

# utils
from .utils import DataDot, DataNum, _color_light_or_dark


    
"""This module aims to implement a basic heatmap functionality by introducing respective classes"""


class Heatmap:
    
    """Class to hold and numeric rectangular data for heatmap plotting via matplotlib's pcolormesh."""

    def __init__(self, 
                 dset:pd.DataFrame, 
                 cluster_rows: bool = False,
                 cluster_cols: bool = False) -> None:
        pass