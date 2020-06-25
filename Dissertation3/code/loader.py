import torch
import sqlalchemy as sql
import pandas as pd
import glob
from os import path, sep
import numpy as np

BASE_LEN = 400

server = "thesis.ca6j6heoraog.eu-central-1.rds.amazonaws.com"
engine = sql.create_engine(f"mysql+pymysql://admin:FphvsYQek4@{server}/thesis_db")

load_signal_query = """
SELECT Concat(participant_id, "_", series_type_id) signal_id,
       data_time,
       data_ordinal,
       data_value
FROM   interpolated_interval_data
WHERE  interpolation_series_id = 4
       AND data_value IS NOT NULL
ORDER BY signal_id, series_type_id, data_ordinal;  
"""

load_cohesion_query = """ 
SELECT participant_id, (cohesion-1)/5 norm_cohesion
FROM participants
WHERE cohesion is not null
"""

def get_data_from_sql(con):
    all_data=pd.read_sql_query(load_signal_query, con)
    cohesion_data = pd.read_sql_query(load_cohesion_query, con)
    all_data.to_pickle('all_data.pkl')
    cohesion_data.to_pickle('cohesion_data.pkl')
    return all_data, cohesion_data

def loader(con, source='disk'):
    if source == 'disk':
        all_data = pd.read_pickle('all_data.pkl')
        cohesion_data = pd.read_pickle('cohesion_data.pkl')
    else: 
        all_data, cohesion_data = get_data_from_sql(con)
    all_data = all_data.set_index("signal_id")
    signal_id_list = list( all_data.index.unique())
    np.random.shuffle(signal_id_list)

    cohesion_dict = {str(int(par)): co for _, (par, co) in cohesion_data.iterrows()}
    
    for sig_id in signal_id_list:
        data = all_data.loc[sig_id].reset_index()
        par = sig_id.split("_")[0]
        coh = cohesion_dict.get(par)
        if not coh:
            #print("not found for {}".format(par))
            continue
        if np.isnan(coh):
            #print("nan found for {}".format(par))
            continue

        min_ord = data.data_ordinal.iloc[0]
        max_ord = data.data_ordinal.iloc[-1]
        data = data.set_index("data_ordinal")
        for i in range(min_ord, max_ord-BASE_LEN,BASE_LEN):
            x = torch.FloatTensor(data.loc[i:i+BASE_LEN-1,"data_value"].values - 0.8) * 10
            yield sig_id, x.reshape(1,1,-1), coh


def connect():
    return engine.connect()
