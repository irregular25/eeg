''' Receive and decode online EEG data '''

import sys
import getopt

from pylsl import StreamInlet, resolve_stream
import nnmodel
import model_nn_transformer

import torch

import mne
from mne.decoding import CSP, Vectorizer, FilterEstimator, Scaler, cross_val_multiscore

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

PATH = "nn_transformer_model2.pt" ## Warning: change to your path

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def update_raw(sample, r):
    ''' concatenate samples as they arrive in a list'''
    for i in range(len(sample)):
        #print(r[i])
        (r[i]).append(sample[i])
    return r


def main(argv):
    min_train = 10
    mod = "nn_transformer" #'svm' # "nn"
    help_string = 'client.py -m <model> -t <min_train>'
    try:
        opts, args = getopt.getopt(argv, "h:m:t", longopts=["model=", "min_train"])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            sys.exit()
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
            ### Receive EEG data and Markers ###
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
        
            ############################## Decoding #################################
            if (len(received_events) >= min_train) & (received_samples % 5000 == 0):
                received_info = mne.create_info(channel_names, sfreq= sample_r, ch_types='eeg')
                received_raw = mne.io.RawArray(received_eeg, received_info, first_sample)
                
                ## Epochs
                #print(received_events)
                # Pick EEG channels, exclude bads
                picks = mne.pick_types(received_info, eeg=True, meg=False, misc=False, exclude='bads')
                #tmin, tmax = -0.5, 2.5 
                tmin, tmax = -1, 4 
                epochs = mne.Epochs(received_raw, received_events, event_id=[2,3], tmin=tmin, tmax=tmax,baseline=None,preload=True,picks=picks)
                                    #,reject=dict(eeg=150e-6)) # rejection threshold too high
                # EEG signals: n_epochs, n_eeg_channels, n_times
                epochs_data = epochs.get_data(copy=False)  
                # labels 0 or 1: hands vs feet
                labels = epochs.events[:, -1] - 2 

                ### Preprocessing steps ###
                ## Apply band-pass filter : reduce high and low frequency noise
                filt = FilterEstimator(epochs.info, 7.0, 30, fir_design='firwin')
                #filt = FilterEstimator(epochs.info, 4.0, 40, fir_design='firwin')

                ## Classical pipeline (not used in the end) ##
                # standardize the data based on channel scales (different from scikit-learn)
                #scaler = Scaler(epochs.info)
                # 2D format for feeding the network
                #vectorizer = Vectorizer()                

                ## CSP: Common Space Pattern
                csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
                
                ## Cross-Validation
                cv = ShuffleSplit(5, test_size=0.2, random_state=42)
                X = epochs_data
                y = labels

                if mod == 'svm':
                    ## Online training and decoding from streaming data ##
                    
                    # SVM model #
                    svm = SVC(kernel='rbf', gamma=0.7, C = 1.0)
                    # Preprocessing #: band-pass filter and csp (before svm)
                    classifiersvm = Pipeline([('filter', filt), ('csp', csp), ('svm', svm)])
                    
                    # Cross-validation to better estimate decoding performance
                    scores_t = cross_val_score(classifiersvm, X, y, scoring=make_scorer(f1_score, average='weighted'), cv=cv, n_jobs=1)
                    print(f" SVM F1 Scores ({scores_t})")
                    std_scores_svm.append(scores_t.std())
                    scores_svm.append(scores_t.mean())
                    iter.append(len(received_events))

                    # Plot F1-score (with std) for Cross-validation as data arrives
                    ax = plt.subplot(111)
                    ax.set_xlabel('Epochs (events)')
                    ax.set_ylabel('Classification f1 score')
                    
                    plt.plot(iter, scores_svm, '-x', color='b',label="Classif. f1 score")
                    ax.plot(iter, scores_svm)
                    hyp_limits = (np.asarray(scores_svm) - np.asarray(std_scores_svm),
                    np.asarray(scores_svm) + np.asarray(std_scores_svm))
                    fill = plt.fill_between(iter, hyp_limits[0], y2=hyp_limits[1], color='b', alpha=0.5)
                    plt.pause(0.01)
                    plt.draw()

                '''if mod == 'nn': # old basic nn model
                    ## NN model ##
                    model = nnmodel.Network().to(device)
                    learning_rate = 1e-3
                    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
                    model.set_optim(opti=optimizer, devi=device)
                    # Preprocessing
                    #classifier_nn = Pipeline([('filter', filt), ('vector', vectorizer),('scaler', scaler), ('nn', model)])
                    classifier_nn = Pipeline([('filter', filt), ('csp', csp), ('nn', model)])
                    scores_t2 = cross_val_multiscore(classifier_nn, X, y, cv=cv, n_jobs=None, scoring=make_scorer(f1_score, average='weighted'))
                    print(f" NN F1 Scores ({scores_t2})")'''

                
                if mod == 'nn_transformer':
                    ## NN Transformer model ##
                    model = model_nn_transformer.TransformerNet().to(device)
                
                    ## This model needs a few iterations to converge, so we will load it here directly ##
                    # Even if the training dataset and training time are repectively bigger and higher, unlike the SVM model,
                    # the decoding is done without any data on the subjects tested.
                    
                    # Model trained on 80 subjects (8-87 arbitrarily) 
                    # (Be careful not to choose one of the subject used for training to avoid bias)
                    model.load_state_dict(torch.load(PATH))

                    # Apply pass-band filter #
                    X = filt.transform(X)
                    
                    learning_rate = 0.0003
                    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
                    model.set_optim(opti=optimizer, devi=device)
                    # Decode data received
                    model.evaluate(X,y)
        
    except KeyboardInterrupt:
        print("User Interrupted")

    finally:
        eeg_inlet.close_stream()
        event_inlet.close_stream()
        print("All receiving streams closed")



if __name__ == '__main__':
    main(sys.argv[1:])
