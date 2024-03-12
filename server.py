''' Server part to send multi-channel EEG data with pylsl'''

import sys
import getopt

import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet, local_clock
import pylsl
import numpy as np
import mne
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage

## Load Data set ##

'''
EEGBCI dataset Characteristics :
- 64-channel EEG data recordings from 109 subjects
- Each subject participated in 14 runs: 
        1: Baseline (eyes open)
        2: Baseline (eyes closed)
        3,7,11: Motor execution: left vs right hand
        4,8,12: Motor Imagery: left vs right hand
        5,9,13: Motor execution: Hands vs Feet
        6,10,14: Motor Imagery: Hands vs Feet
- Recordings were made using the BCI2000 system.

for further details, see: https://www.physionet.org/content/eegmmidb/1.0.0/
'''

## NB: For time issues, here we compute the motor imagery data of 1 subject
# The data of all subjects can be retrieved like this (commented section).
'''subject_list = [i for i in range(1,110)]
all_data = []
for subject in subject_list:
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    all_data.append(raw)'''

def main():
    ''' send the raw EEG data and the associated events with pylsl to the client
    of the subject
    '''
    runs = [6, 10, 14]  # motor imagery: hands vs feet
    subject=1
    
    # Load mne dataset
    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    # set suitable format for channel names
    eegbci.standardize(raw)  
    # standard montage
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    
    # Retrieve Events
    events, event_id = mne.events_from_annotations(raw)
    event_info = pylsl.StreamInfo(name="EventMarkers", type="Markers", channel_count=1, 
                        nominal_srate=0, channel_format="string", source_id="eve")
    event_outlet = pylsl.StreamOutlet(event_info)

    # Retrieve EEG data
    raw = raw.pick(picks=["eeg"])
    eeg_info = pylsl.StreamInfo(name='MNE', type="EEG",
                                channel_count=raw.info["nchan"],
                                nominal_srate=raw.info["sfreq"],
                                channel_format='float32', source_id='eeg')
    # Meta Data
    ch_names = raw.info["ch_names"]
    chns = eeg_info.desc().append_child("channels")
    for label in ch_names:
        ch = chns.append_child("channel")
        ch.append_child_value("label", label)
    total_samples = eeg_info.desc().append_child("n_times")
    total_samples.append_child_value("nb", str(raw.n_times))
    first_samp = eeg_info.desc().append_child("first_samp")
    first_samp.append_child_value("val", str(raw.first_samp))

    eeg_outlet = pylsl.StreamOutlet(eeg_info)

    ## Send Data ##
    print("sending data...")
    try :
        counter = 0
        event_counter = 0
        index_first_sample = raw.first_samp
        index_last_sample = raw.last_samp
        pos_last_sample = raw.last_samp - raw.first_samp
        pos_last_event = len(events) - 1
    
        while counter <= pos_last_sample :

            # get sample for all channel
            eeg_sample = raw[:,counter][0].ravel()

            # get event marker if present
            if events[event_counter][0] == (index_first_sample + counter) :
                #print(counter)
                event_marker = str(events[event_counter][2])
            
                event_outlet.push_sample([event_marker])
                # go to next sample or reset if all markers seen
                event_counter = 0 if event_counter == pos_last_event else event_counter + 1

            eeg_outlet.push_sample(eeg_sample)
            # go to next sample
            counter += 1

            # time value can be adjusted to better synchronize streams
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("User interrupted the stream")

    # release resources
    print("Deleting unused outlets")
    del eeg_outlet
    del event_outlet

if __name__ == '__main__':
    main()    
    

