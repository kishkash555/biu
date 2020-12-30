import numpy as np
import pandas as pd

corr_to_angdist = lambda rho: np.arccos(rho)/np.pi

def angular_dist_in_rows(df):
    dist_df= df.corr().melt(var_name='distance').apply(corr_to_angdist)
    dist_df.set_
def angular_distance_matrix(df,level=None):
    """
    calculate the angular distance metric between columns of a dataframe
    level refers to multiindex, limiting the pairing to columns at lower levels
    """

    
    if level is not None:
        for sub_df in df.groupby(level=level,axis=1):
