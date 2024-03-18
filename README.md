# Test code

Here is the repository for my project about streaming EEG data from a server to a client with pylsl and then decoding it.
The required python libraries are listed in requirements.txt

## Dataset

For this project, the EEGBCI MNE dataset was chosen as it consists of motor imagery EEG data.
109 subjects performed different motor/imagery tasks while 64-channel EEG were recorded using the BCI2000 system.
Each subject performed 14 experimental runs:
  - 1: Baseline (eyes open)
  - 2: Baseline (eyes closed)
  - 3,7,11: Motor execution: left vs right hand
  - 4,8,12: Motor Imagery: left vs right hand
  - 5,9,13: Motor execution: Hands vs Feet
  - 6,10,14: Motor Imagery: Hands vs Feet

For this project, the focus was on the Motor Imagery EEG data and only the runs 6,10,14 were used. 
In these runs, the "Hands" task involves closing both fists, while the "left vs right" task consists of closing only one.
To avoid any bias, the 4,8,12 runs were thus not considered in this project.

For further information about this dataset, see https://www.physionet.org/content/eegmmidb/1.0.0/

The 64 channels of these dataset are shown on the figure below.

![plot](https://github.com/irregular25/eeg/edit/main/eegbci_channels.png?raw=true)

## Run

To run this project, execute in this order: 
- python client.py
- python server.py

Some extra arguments are available for this 2 files.
For the server, you can indicate which subject (1-109) to choose for the data : python server.py -s "subject number"

For the client, you can indicate:
- the model for decoding ('svm' or nn transformer model 'nn_transformer', default is 'svm'): python client.py -m "model name"
- the minimum number of EEG epochs data to collect before training: python client.py -t "minimum number"

You can also add the argument -h for both files to see how to pass arguments.

## Models

For every model, a pass-band filter is first applied on the EEG data to remove low and high frequency noise.

### SVM

The SVM (Support Vector Machine) model can be used for decoding the streaming EEG data online and is trained as the data arrives.
To prepare the data and reduce their dimension, the CSP (Common Space Pattern) algorithm, a well-known algorithm for feature extraction in EEG Motor-Imagery data is first applied.

### Transformer model

The Transformer model (CNN and transformer encoder architecture) can be loaded and used for decoding online. 
Even if it is trained on a larger dataset (80 subjects, cf model_nn_transformer.py), unlike the SVM model,
the decoding is done without any data on the subject tested.

For reference, a Transformer Only model (Without CNN) is also given (model_transformer_only.py) even though it doesn't seem to converge. 
This shows the benefits of using CNN layers for dimension reduction and extracting local features (spatio-temporal) from EEG data.

### S4 Model (Update)
To investigate the potential of SSM models on EEG data, a S4 model has also been implemented and can be run in model_s4.py.

Hugues
