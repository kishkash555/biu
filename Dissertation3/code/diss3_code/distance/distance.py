import numpy as np
import pandas as pd

corr_to_angdist = lambda rho: np.arccos(rho)/np.pi

def angular_dist_in_rows(df):
    dist_df= df\
        .corr()\
        .apply(corr_to_angdist)\
        .melt(col_level=1,ignore_index=False,var_name='other',value_name='dist')\
        .reset_index()\
        .assign(
            joined=lambda d: d.apply(
                lambda r: np.nan if r.level_1==r.other 
                    else '-'.join(sorted([r.level_1,r.other])), 
                axis=1
                ) 
            )\
            .dropna()\
            .drop_duplicates('joined',keep='last')\
            .drop(columns='joined')

    return dist_df    
      #  dist_df.set_

def angular_distance_matrix(df,level=None):
    """
    calculate the angular distance metric between columns of a dataframe
    level refers to multiindex, limiting the pairing to columns at lower levels
    """

    angular_dist_all = []
    if level is not None:
        for _, sub_df in df.groupby(level=level,axis=1):
            angular_dist_all.append(angular_dist_in_rows(sub_df))
    
    angular_dist_all = pd.concat(angular_dist_all)
    return angular_dist_all
    
