import numpy as np
import pandas as pd
from scipy.signal import remez, oaconvolve, savgol_filter

# import sys
# from os import path 
# sys.path.append(path.abspath('../..'))

import smooth_component_analysis as sca


hrv_lf_h = remez(25, [0., 0.1, 0.15, 1.],[1., 0.],fs =2.)
hrv_rms_filter = lambda y: savgol_filter(y, 11, 2)

filter_hrv_lf = lambda y: oaconvolve(y, hrv_lf_h, 'valid')
ecg_lf_filter = sca.dataframe_filter(
    filter_hrv_lf, 
    naming_func=lambda s: s.replace('_raw','_lf'),
    pad_output=[12,12]
    )

ecg_rms_filter = sca.dataframe_filter(
    hrv_rms_filter,
    naming_func=lambda s: s.replace('_hf','_rms'),
    pad_output=None
)

INTERPOLATION_SERIES_ID_FOR_HRV = 3
# 1005 does not have cohesions so removed for now
valid_ecg_groups = [1004,  1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014,
       1016, 1019, 1020, 1022, 1024, 1025, 1026, 1027, 1028, 1029, 1030,
       1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041,
       1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051]
sample_groups = [1001, 1004, 1006]
session_codes = {'BL': 1, 'I':2, 'BLE': 3, 'FS': 4}


def get_hrv_data_for_groups(group_ids, session_type, lf=True, hf=True, rms=True):
    lf_df, hf_df, rms_df = None, None, None

    raw_df = sca.get_signals(
        group_ids, 
        session_codes.get(session_type, session_type),
        INTERPOLATION_SERIES_ID_FOR_HRV,
        "{}_{}_raw".format,
        standardize=True)
    
    ret = raw_df
    if lf or hf or rms:
        lf_df = ecg_lf_filter.forward(raw_df, return_residual = hf, resid_naming_func = lambda s: s.replace('_raw','_hf'))
        if hf:
            lf_df, hf_df = lf_df # returned a 2-tuple, split accordingly
            ret = ret.join(lf_df).join(hf_df)
        else:
            ret.join(lf)
        if rms:
            rms_df = ecg_rms_filter.forward(hf_df.dropna()**2)
            ret = ret.join(rms_df, how='left')
    
    return ret


if __name__ == "__main__":
    print(__file__)
    works = get_hrv_data_for_groups(valid_ecg_groups, 2)
    a=5