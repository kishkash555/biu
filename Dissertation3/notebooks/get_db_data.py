import sqlalchemy as sql
import numpy as np

signal_sql = sql.text("""
SELECT data_time, data_value
FROM interpolated_interval_data
INNER JOIN group_series_interpolation_interval USING
    (series_type_id, interpolation_series_id, group_id)
WHERE participant_id = :par
AND series_type_id = :series_type
AND interpolation_series_id = :interp_type
AND data_value is not null
AND data_time >= interval_start
ORDER BY data_time
""")


group_signal_sql = sql.text("""
SELECT participant_id, data_time, data_value
FROM interpolated_interval_data
INNER JOIN group_series_interpolation_interval USING
    (series_type_id, interpolation_series_id, group_id)
WHERE group_id = :grp
AND series_type_id = :series_type
AND interpolation_series_id = :interp_type
AND data_value is not null
AND data_time >= interval_start
AND data_time <= interval_end
ORDER BY participant_id, data_time
""")


# partid_query = """
# SELECT DISTINCT group_id, participant_id
# FROM raw_interval_data
# """

partid_query = lambda ret_cohesion: ''.join([
    "SELECT p.group_id, p.participant_id", 
    ret_cohesion and ", cohesion" or "" ,
    """
    FROM participants p
    """,
    ret_cohesion and "WHERE cohesion IS NOT NULL" or "" 
    ])

class get_db_data:
    def __init__(self):
        self.con = get_db_data.connect_aws()
        self.participants = None    


    @staticmethod
    def connect_aws():
        server = "thesis.ca6j6heoraog.eu-central-1.rds.amazonaws.com"
        engine = sql.create_engine(f"mysql+pymysql://admin:FphvsYQek4@{server}/thesis_db")
        con = engine.connect()
        return con


    def get_signal_by_series_type_and_interp_type(self, participants, series_type, interp_type):
        if type(participants)==int and participants > 1000: # a groupid
            participants = self.grouped_participants[participants]
        for par in participants:
            if par < 1000:
                data = self.con.execute(signal_sql, par=par, series_type=series_type, interp_type=interp_type)
                yield par, np.array(data.fetchall()).astype(float)
            else: # actually a group:
                raise NotImplementedError(
                    "This functionality moved to get_group_signals_by_series_type_and_interp_type."
                )
            
    def get_group_signals_by_series_type_and_interp_type(self, group_ids, series_type, interp_type):
        for par in group_ids:
            data = self.con.execute(group_signal_sql, grp=par, series_type=series_type, interp_type=interp_type)
            yield np.array(data.fetchall()).astype(float)



    def get_participants(self,return_cohesions=False,grouped=False):
        if return_cohesions and grouped:
            raise ValueError("Function does not support True, True input combination") 
        participants = list(self.con.execute(partid_query(return_cohesions)))
        self.participants = participants
        pars_tuples = {}
        if not return_cohesions:
            return participants
        for g, p, _ in participants:
            pars_tuples[g] = pars_tuples.get(g,[])  + [p]
        self.grouped_participants = pars_tuples
        return pars_tuples if grouped else participants    
    
    def close(self):
        self.con.close()