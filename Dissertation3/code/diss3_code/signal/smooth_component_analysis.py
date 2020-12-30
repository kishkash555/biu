import sys
from os import path 
sys.path.append(
    path.abspath(
        path.join(__file__,
        '../../database')
        )
)
print(sys.path)
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
        self.lf_filter = lambda x: savgol_filter(x, self.f1_length,1)
        self.rms_filter = lambda x: np.sqrt(savgol_filter(x**2, self.f2_length,1,mode="mirror"))
        self.debug = False

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
            sg = self.lf_filter if not self.debug else savgol_filter(y,self.f1_length,1)
            resid = y - sg
            rms = self.rms_filter(resid) if not self.debug else \
                np.sqrt(savgol_filter(resid**2,self.f2_length,1,mode="mirror"))
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


class dataframe_filter:
    def __init__(self, filter_func, naming_func, pad_output, *filter_args, **filter_kwargs) -> None:
        """
        caveat: 
        - the filter func must return the same number of samples as the input.
        Therefore, if your filter function returns a timeseries shorter than the input,
        pad it as necessary with np.nan before calling __init__
        or indicate the number of samples that are going to be dropped on either side 
        using the pad_output argument
        filter_func: callable: numpy array -> numpy array
        naming_func: callable: string -> string
        pad_output: either None or [prepend_size, postpend_size]
        """
        super().__init__()
        self.filter_func = filter_func
        self.naming_func = naming_func
        self.pad = [0,0] if pad_output is None else pad_output
        self.filter_args = filter_args
        self.kwargs = filter_kwargs

    def forward(self, input_df, return_residual=False, resid_naming_func = None):
        # initialize a dataframe with same shape as input, all values nan
        df_data = np.nan*np.ones_like(input_df)
        ls,le = (self.pad[0], df_data.shape[0]-self.pad[1])
        
        # calculate start- and end- index of filtered data

        # apply the filter and assign to the correct place in the dataframe
        df_data[ls:le,:] = input_df.apply(self.filter_func,
            axis=0,
            args=self.filter_args,
            **self.kwargs
            ).values

        df = pd.DataFrame(data=df_data, 
            index=input_df.index, 
            columns = [self.naming_func(col) for col in input_df.columns])
        
        if return_residual:
            resid_naming_func = resid_naming_func if resid_naming_func is not None else lambda s: s
            resid = pd.DataFrame(
                columns = [resid_naming_func(col) for col in input_df.columns], 
                index= input_df.index, 
                data = input_df.values-df.values
                )
            return df, resid
        return df

def get_signals(group_ids, series_type, interp_type, col_name_func=None, standardize=True):
    """
    returns a dataframe with all the signals from the specified series_type and interp_type
    the x axis (time frame) is inner-joined so any missing samples are dropped
    """
    df = pd.DataFrame()
    how = 'right'
#    stdzr = zscore if standardize else lambda x: x
    for gid in group_ids:
        for par_id, signal in yield_group_signals(gid,series_type, interp_type):
            if signal.shape[0]< 5:
                print("participant {} has not data".format(par_id))
                continue
            df = df.join(
                pd.DataFrame(
                    index=signal[:,0],
                    data=signal[:,1],
                    columns=[col_name_func(gid, par_id)],
                    ),
                how=how
                )
            how='inner' # after the first go, everything else should be inner-join
    if standardize:
        # discovered weird behavior that can be corrected by setting ddof=1
        df = df.apply(zscore) 
    return df


