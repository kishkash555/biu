select data_ordinal, a_start_time, a_end_time, event_start_time, event_end_time, cast( 100*(event_end_time-event_start_time)/(a_end_time-a_start_time) as decimal(4,2)) p, a_value, b_value
from (
select a.data_ordinal, a.start_time a_start_time, a.end_time a_end_time, greatest(a.start_time,b.start_time) event_start_time, least(a.end_time, b.end_time) event_end_time, a.data_time_diff a_value, b.data_time_diff b_value
from 
( select data_ordinal, data_time-data_time_diff start_time, data_time end_time, data_time_diff
from raw_interval_data
where participant_id = 164
and series_type_id= 1 ) a
inner join
(select data_time-data_time_diff start_time, data_time end_time, data_time_diff
from raw_interval_data
where participant_id = 165
and series_type_id = 1 ) b
on (a.start_time <= b.start_time and a.end_time >= b.start_time or a.start_time <= b.end_time and a.end_time >= b.end_time)
) c