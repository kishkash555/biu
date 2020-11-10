

insert into group_series_interpolation_interval
select series_type_id, interpolation_series_id, group_id, max(min_time) interval_start, min(max_time) interval_end
from (
	select series_type_id, interpolation_series_id, group_id, 
		participant_id, 
		min(data_time) min_time, 
		max(data_time) max_time
	from interpolated_interval_data
	where interpolation_series_id=3
	and data_value is not null
	and series_type_id=4
	group by series_type_id, interpolation_series_id, group_id,participant_id
) a
group by series_type_id, interpolation_series_id, group_id