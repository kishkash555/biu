import music21 as m2
import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import collections  as mc
import os.path as path

def load_midi(midi_path):
    """
    example midi path: midi_path = '../data/MIDI-Freestyle/Freestyle_1044.mid'
    """
    mf = m2.midi.MidiFile()
    mf.open(midi_path)
    mf.read()
    mf.close()
    return mf


def populate_music_dictionary(midi_files):
    all_ticks = {}
    for mf in midi_files:
        group_num = path.basename(mf).split('_')[1].split('.')[0]
        session = path.basename(mf).split('_')[0]
        group_ticks = get_drumbeat_ticks_for_midi(mf)
        for track, ticks in group_ticks.items():
            print((group_num, session, track))
            assert (group_num, session, track) not in all_ticks
            all_ticks[(group_num, session, track)]= [t[0] for t in ticks if t[1]-t[0]>10] 
    return all_ticks

def to_dense(t,max_len=None):
    max_len = max_len or t[-1]
    ret = np.zeros(max_len)
    max_index_to_include = np.sum(t<max_len).astype(int)
    ret[t[:max_index_to_include]]=1.
    return ret

def get_drumbeat_ticks_for_midi(midi_path):
    mf = load_midi(midi_path)
    ev = mf.tracks[0].events
    df = pd.DataFrame(data=[e.__dict__ for e in ev])
    df.insert(2,'type_str',df.type.map(lambda t: str(t).split('.')[-1]))
    df.insert(4,'time_abs',df.time.cumsum())
    #print(mf.ticksPerQuarterNote)
    all_events = {}
    open_events = {}
    for row in df[df.type_str.isin(['NOTE_ON','NOTE_OFF'])].itertuples():
        if row.type_str=='NOTE_ON':
            open_events[row.parameter1]=row.time_abs
        else:
            all_events[row.parameter1]=all_events.get(row.parameter1,[])+[(open_events[row.parameter1], row.time_abs)]

    return all_events
   

def get_files_for_session(midi_path, session='freestyle'):
    if session.lower() == 'freestyle':
        midi_files = glob.glob(path.join(midi_path,'Freestyle_*.mid'))
        group_number = lambda fname: int(fname.split('_')[1][:4])
        file_list = sorted([file for file in midi_files if group_number(file) <= 1013 or group_number(file) >=1042])
        return file_list
    else:
        raise NotImplementedError('only freestyle session is currently supported')

class group_events:
    def __init__(self, group_id, events) -> None:
        self.group_id = group_id
        self.events = events
    
    def get_durations(self) -> np.array:
        return { track_id : ev.get_durations() for track_id, ev in self.events.items() }

    def get_intervals(self) -> np.array:
        return { track_id: ev.get_intervals() for track_id, ev in self.events.items() }

    def get_counts(self) -> dict:
        return { track_id: len(ev) for track_id, ev in self.events.items() }

    def filter(self, filter_obj):
        return group_events(self.group_id, { 
            track_id: ev.filter(filter_obj) for track_id, ev in self.events.items() 
            })

class drummer_events:
    def __init__(self, event_list):
        self.event_list = list(event_list)
    
    def get_durations(self):
        return np.diff(self.event_list)
    
    def get_intervals(self):
        return np.diff(self.event_list, axis=0)[:,0]

    def filter(self, filter_obj):
        mask = filter_obj(self.event_list)
        return [a for a, m in zip(self.event_list, mask) if m]

    def __len__(self):
        return len(self.event_list)

    def __iter__(self):
        return iter(self.event_list)


class event_filter:
    def __init__(self) -> None:
        self.method = None
        self.params = {}

    @classmethod
    def duration(cls, min_duration):
        ev = event_filter()
        ev.method = ev._duration
        ev.params['min_duration'] = min_duration
        
    def __call__(self, event_list):
        return self.method(event_list)
    
    def _duration(self, event_list):
        min_duration = self.params['min_duration']
        return [a for a in event_list if a[1]-a[0] >= min_duration]

        return event_filter(type = 'duration')
        # durations = events_obj.get_durations()

        event_dict_filtered = { 
            track_id: [v for v in ev if v[1]-v[0] >= min_duration ] 
            for track_id, ev in events_obj.events.items()
            }
        return midi_events(events_obj.group_id, event_dict_filtered)