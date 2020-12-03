import loader
import pandas as pd
import sqlalchemy as sql
import psd


def load_data(con):
    data_loader = loader.loader(con,source='disk', base_len=880)
    data = pd.DataFrame(data_loader.generator(group_id=True)).drop_duplicates(subset='sig_id', keep='first')
    data['fft'] = data.signal.map(lambda x: psd.psd(x.numpy().squeeze(),0.25))
    return data


if __name__ == "__main__":
    #server = "thesis.ca6j6heoraog.eu-central-1.rds.amazonaws.com"
    #engine = sql.create_engine(f"mysql+pymysql://admin:FphvsYQek4@{server}/thesis_db")
    #con = engine.connect()
    load_data(None)