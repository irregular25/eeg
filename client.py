''' Receive and decode online EEG data '''

import sys
import getopt

from pylsl import StreamInlet, resolve_stream
import server
import nnmodel

import mne
from mne.decoding import CSP

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mne.decoding import Vectorizer, FilterEstimator

def update_raw(sample, r):
    ''' concatenate samples as they arrive in a list'''
    for i in range(len(sample)):
        #print(r[i])
        (r[i]).append(sample[i])
    return r


def main(argv):
    subject= 1
    min_train = 10
    mod = 'svm'
    help_string = 'client.py -m <model> -s <subject>'
    try:
        opts, args = getopt.getopt(argv, "h:m:s:t", longopts=["model=", "subject=", "min_train"])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            sys.exit()
        elif opt in ("-s", "--subject"):
            subject = int(arg)
        elif opt in ("-t", "--min_train"):
            min_train = float(arg)
        elif opt in ("-m", "--model"):
            mod = str(arg)

    # Look for Streams
    print("looking for an EEG stream...")
    eeg_streams = resolve_stream('type', 'EEG')
    eeg_inlet = StreamInlet(eeg_streams[0])

    print("looking for an Event stream...")
    event_streams = resolve_stream('type', 'Markers')
    event_inlet = StreamInlet(event_streams[0])

    # Receive Meta Data
    sample_r = eeg_inlet.info().nominal_srate()
    nb_channels = eeg_inlet.info().channel_count()
    first_sample = int(eeg_inlet.info().desc().child("first_samp").child_value("val"))
    channel_n = eeg_inlet.info().desc().child("channels").child("channel")
    channel_names = []
    for k in range(nb_channels):
        channel_names.append(channel_n.child_value("label"))
        channel_n = channel_n.next_sibling()

    desired_samples = int(eeg_inlet.info().desc().child("n_times").child_value("nb"))
    received_samples = 0

    received_eeg = []
    received_events = []

    # variables for plotting results
    iter = []
    scores_svm = []
    std_scores_svm = []

    try:
        while received_samples < desired_samples :
            # Receive EEG data and Markers
            sample, timestamp = eeg_inlet.pull_sample()
            event_markers, event_timestamp = event_inlet.pull_sample(timeout=0)
        
            # Concatenate EEG data
            if len(received_eeg) == 0:
                received_eeg = np.array(sample).reshape(nb_channels,1)
                received_eeg = received_eeg.tolist()
            else :
                received_eeg = update_raw(sample, received_eeg)
        
            # Concatenate Markers
            if event_markers is not None :
                e = int(event_markers[0])
                event = [first_sample + received_samples, 0, e]
                received_events.append(event)

            # Keep track of data received
            received_samples += 1
            if (received_samples % 1000 == 0) :
                print(f"Received EEG data ({received_samples}/{desired_samples})")
        
            ## Decoding
            if (len(received_events) >= min_train) & (received_samples % 5000 == 0):
                received_info = mne.create_info(channel_names, sfreq= sample_r, ch_types='eeg')
                received_raw = mne.io.RawArray(received_eeg, received_info, first_sample)
                
                # Epochs
                print(received_events)
                # Pick EEG channels, exclude bads
                picks = mne.pick_types(received_info, eeg=True, meg=False, misc=False, exclude='bads')
                epochs = mne.Epochs(received_raw, received_events, event_id=[2,3], tmin=-0.5, tmax=2.5,baseline=None,preload=True,picks=picks)
                                    #,reject=dict(eeg=150e-6))
                X = epochs.get_data(copy=False)  # EEG signals: n_epochs, n_eeg_channels, n_times
                y = epochs.events[:, 2]  # target: 
                # Apply band-pass filter
                filt = FilterEstimator(epochs.info, 7.0, 30, fir_design='firwin')
                scaler = StandardScaler()
                vectorizer = Vectorizer()
                # labels 0 or 1:
                labels = epochs.events[:, -1] - 2

                # CSP
                csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

                # SVM model
                svm = SVC(kernel='rbf', gamma=0.7, C = 1.0)

                classifier0 = Pipeline([('filter', filt), ('vector', vectorizer),
                              ('scaler', scaler), ('svm', svm)])
                classifiersvm = Pipeline([('filter', filt), ('csp', csp), ('svm', svm)])
                
                # Cross-Validation
                epochs_data = epochs.get_data(copy=False) # EEG signals: n_epochs, n_eeg_channels, n_times
                cv = ShuffleSplit(5, test_size=0.2, random_state=42)
                X = epochs_data
                y = labels

                if mod == 'svm':
                    
                    scores_t = cross_val_score(classifiersvm, X, y, scoring=make_scorer(f1_score, average='weighted'), cv=cv, n_jobs=1)
                    print(f"F1 Scores ({scores_t})")
                    std_scores_svm.append(scores_t.std())
                    scores_svm.append(scores_t.mean())
                    iter.append(len(received_events))

                    # Plot F1-score
                    ax = plt.subplot(111)
                    ax.set_xlabel('Epochs (events)')
                    ax.set_ylabel('Classification f1 score')
                    
                    plt.plot(iter[-2:], scores_svm[-2:], '-x', color='b',label="Classif. f1 score")
                    ax.plot(iter[-1], scores_svm[-1])
                    hyp_limits = (np.asarray(scores_svm) - np.asarray(std_scores_svm),
                    np.asarray(scores_svm) + np.asarray(std_scores_svm))
                    fill = plt.fill_between(iter, hyp_limits[0], y2=hyp_limits[1], color='b', alpha=0.5)
                    plt.pause(0.01)
                    plt.draw()
                    fill.remove()  # Remove old fill area


        # Final figure
        plt.fill_between(iter, hyp_limits[0], y2=hyp_limits[1], color='b', alpha=0.5)
        plt.draw()
        
    except KeyboardInterrupt:
        print("User Interrupted")

    finally:
        eeg_inlet.close_stream()
        event_inlet.close_stream()
        print("All receiving streams closed")



if __name__ == '__main__':
    main(sys.argv[1:])
