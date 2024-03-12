# Test code

Here is the repository for my project streaming EEG data from a server to a client with pylsl and then decoding it.
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

## Run

To run this project, execute in this order: 
- python client.py
- python server.py

Some extra arguments are available for this 2 files.
For the server, you can indicate which subject (1-109) to choose for the data : python server.py -s "subject number"

For the client, you can indicate:
- the model for decoding ('svm' or transformer model 'nn', default is 'svm'): python client.py -m "model name"
- the minimum number of EEG epochs data to collect before training: python client.py -t "minimum number"

You can also add the argument -h for both files to see how to pass arguments.

Hugues
