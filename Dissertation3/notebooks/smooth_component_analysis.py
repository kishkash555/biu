
import get_db_data as gdd
from scipy.signal import savgol_filter
from scipy.stats import zscore
import numpy as np
import pandas as pd

data = gdd.get_db_data()
pars = data.get_participants(True)

pars_tuples={}
for g, p, _ in pars:
    pars_tuples[g] = pars_tuples.get(g,[])  + [p]


gm_orth = lambda v,u: v - u * (v@u)/(u@u) 

def yield_group_signals(group_id, series_type, interp_type):
    return data.get_signal_by_series_type_and_interp_type(group_id,series_type,interp_type)

class savitzky_golay:
    def __init__(self, group_ids, f1_length, f2_length,series_type, interp_type):
        self.group_ids = group_ids
        self.f1_length = f1_length
        self.f2_length = f2_length
        self.series_type = series_type
        self.interp_type = interp_type
        self.df = None

    def group_signal_components(self, group_id, return_x=False):
        Raw = "Raw_{}".format
        SG = "Sg_{}".format
        Resid = "resid_{}".format
        RMS = "rms_{}".format
        RMSO = "rmso_{}".format
        signals = {}
        
        for par_id, sig in yield_group_signals(group_id, self.series_type, self.interp_type):
            x = sig[:,0]
            y = zscore(sig[:,1])
            sg = savgol_filter(y,self.f1_length,1)
            resid = y - sg
            rms = np.sqrt(savgol_filter(resid**2,self.f2_length,1,mode="mirror"))
            rmso = gm_orth(rms,sg)
            signals.update({
                Raw(par_id): y,
                SG(par_id): sg,
                Resid(par_id): resid,
                RMS(par_id): rms,
                RMSO(par_id): rmso
            })
        return (signals, x) if return_x else signals

    def component_correlations_within_each_group(self):
        """
        THIS FUNCTION IS DEPRECATED IN FAVOR OF YIELD_CROSSGROUP_CORR
        ------------------------------------------------------------
        returns a dictionary.
        keys: group_ids
        values: correlation dataframes 
        of the correlation among components within each group
        """
        cr = {}
        for gid in self.group_ids:
            sig = self.group_signal_components(gid)
            df = pd.DataFrame(sig)
            df=df[sorted(df.columns)]
            cr[gid] = df.corr()
        return cr


    def yield_crossgroup_corr(self, same_group=False, col_type="rmso_",yield_with_colnames = False):
        df = self.df if self.df is not None else self.get_aligned_signal_componets()
        cdf = df.corr()
        cols = [c for c in cdf.columns if c[1].startswith(col_type)]
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                if (cols[i][0] == cols[j][0])==same_group: # different groups
                    print(cols[i], cols[j])
                    if yield_with_colnames:
                        yield ((cols[i], cols[j]), cdf.loc[cols[i],cols[j]])
                    else:
                        yield cdf.loc[cols[i],cols[j]]

    def get_aligned_signal_componets(self):
        min_x, max_x = [], []
        all_signal_components = [self.group_signal_components(gid,True) 
            for gid in self.group_ids]
        min_x = max(a[1][0] for a in all_signal_components)
        max_x = min(a[1][-1] for a in all_signal_components)
        print("x range: {} {}".format(min_x, max_x))
        # create a dataframe with decomposed signals from all groups 
        df = pd.DataFrame()
        for gid, (signals_dict, x) in zip(self.group_ids, all_signal_components):
            for sig in signals_dict.keys():
                df[(gid, sig)] = signals_dict[sig][(min_x <= x) & (x <= max_x)]

        df.columns = pd.MultiIndex.from_tuples(df.columns)
        self.df = df
        return df
