import torch
import sqlalchemy as sql
import pandas as pd
import glob
from os import path, sep
import numpy as np
from sklearn.model_selection import train_test_split
from collections import namedtuple
from scipy.stats import zscore

signal_rec = namedtuple("signal_rec","group_id,sig_id,signal,cohesion".split(","))

server = "thesis.ca6j6heoraog.eu-central-1.rds.amazonaws.com"
engine = sql.create_engine(f"mysql+pymysql://admin:FphvsYQek4@{server}/thesis_db")

signal_query_version = 2

load_signal_query_template = """
SELECT Concat(participant_id, "_", series_type_id) signal_id,
       data_time,
       data_ordinal,
       data_value
FROM   interpolated_interval_data
WHERE  interpolation_series_id = {}
       AND data_value IS NOT NULL
       {}
ORDER BY signal_id, series_type_id, data_ordinal;  
"""

load_signal_query = load_signal_query_template.format(*
    signal_query_version==1 and (4,"") or 
    signal_query_version==2 and (3,"AND series_type_id = 2")
    )


load_cohesion_query = """ 
SELECT participant_id, group_id, (cohesion-1)/5 norm_cohesion
FROM participants
WHERE cohesion is not null
"""

def get_data_from_sql(con):
    all_data=pd.read_sql_query(load_signal_query, con)
    cohesion_data = pd.read_sql_query(load_cohesion_query, con)
    all_data.to_pickle('all_data.pkl')
    cohesion_data.to_pickle('cohesion_data.pkl')
    return all_data, cohesion_data

class loader:
    """
    This class handles getting the signal data and providing it to any consumer class
    as a generator.
    The generator handles slicing signals and shuffling.
    The data may also be split to train and validation
    """
    def __init__(self, con, source='disk', split=False, base_len=400):
        if source == 'disk':
            all_data = pd.read_pickle('all_data.pkl')
            cohesion_data = pd.read_pickle('cohesion_data.pkl')
        else: 
            all_data, cohesion_data = get_data_from_sql(con)
        
        all_data["pars"] = all_data.signal_id.map(lambda x: x.split("_")[0])
        pars = all_data.pars.unique()
        if split:
            train_pars, validation_pars = train_test_split(pars, test_size=0.2,shuffle=False)
        else:
            train_pars = pars
            validation_pars = []
            
        all_data["learn_group"]=all_data.pars.map(
            lambda x: x in set(train_pars) and "train" or "validation"
            )
            
        #self.all_data = all_data
        self.cohesion_dict = { 
            str(int(r.participant_id)): r.norm_cohesion 
            for _, r in cohesion_data.iterrows()
            }
        
        self.group_id_dict = { 
            str(int(r.participant_id)): int(r.group_id) 
            for _, r in cohesion_data.iterrows()
            }
        self.all_data = all_data.set_index("signal_id")
        self.base_len = base_len

    def generator(self, training=True, group_id=False, standardize=False):
        BASE_LEN = self.base_len
        learn_group = training and "train" or "validation"
        curr_df = self.all_data[self.all_data.learn_group==learn_group]
        
        signal_id_list = list(curr_df.index.unique())
        if not group_id:
            np.random.shuffle(signal_id_list)

        for sig_id in signal_id_list:
            if sig_id=='119_3':
                continue
            data = self.all_data.loc[sig_id].reset_index()
            par = sig_id.split("_")[0]
            coh = self.cohesion_dict.get(par)
            if not coh:
                #print("not found for {}".format(par))
                continue
            if np.isnan(coh):
                #print("nan found for {}".format(par))
                continue

            min_ord = data.data_ordinal.iloc[0]
            max_ord = data.data_ordinal.iloc[-1]
            data = data.set_index("data_ordinal")
            if standardize:
                data["data_value"] = zscore(data["data_value"])
            y = []
            for i in range(min_ord, max_ord-BASE_LEN,BASE_LEN):
                x = data.loc[i:i+BASE_LEN-1,"data_value"].values
                x = x- x.mean()
                x = torch.FloatTensor(x)
                if group_id:
                    y.append((
                        signal_rec(
                        self.group_id_dict[sig_id.split("_")[0]],
                        sig_id, 
                        x.reshape(1,1,-1), 
                        coh)
                    ))
                else:
                    y.append((sig_id, x.reshape(1,1,-1), coh))
            yield y


def connect():
    return engine.connect()


if __name__ == "__main__":
    ld = loader(connect(),'sql',base_len=50)

    a = ld.generator()

    sig_id, x, coh = next(a)
    print(x.shape)
    1
