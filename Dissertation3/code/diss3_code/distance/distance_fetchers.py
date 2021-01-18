from os import path
import sys
sys.path.append(path.abspath(path.join(__file__,'..','..','signal')))
import signal_fetchers as sf
import pandas as pd
# from .. import signal_fetchers as sf
import distance
rearrange_header = lambda h: '_'.join(h.split('_')[i] for i in [0,2,1])


def all_ecg_distances_by_session(session=2, groups=sf.valid_ecg_groups):
    """
    get a long-form df for distances in each group, for a session
    session is a number 1-4 or their name: {'BL': 1, 'I':2, 'BLE': 3, 'FS': 4}(see signal_fetchers.session_codes)
    """
    df = sf.get_hrv_data_for_groups(sf.valid_ecg_groups, session)
  
    df.columns = pd.MultiIndex.from_tuples([tuple(rearrange_header(s).rsplit('_',1)) for s in df.columns ])
    dist_mat = distance.angular_distance_matrix(df, level=0)
    return dist_mat


def get_all_sessions_all_groups():
    sessions = ['BL','I', 'BLE', 'FS']
    all_sessions_all_groups = pd.concat([all_ecg_distances_by_session(s).assign(session=s) for s in sessions])
    return all_sessions_all_groups


if __name__ == "__main__":
    asag = get_all_sessions_all_groups()
    a=5