''' Neural Network with CNN and Transformer encoder'''
''' issue: seems to be overfitting'''

import numpy as np
import mne
import pylsl
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from mne.decoding import Vectorizer, FilterEstimator, Scaler,cross_val_multiscore

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from pylab import rcParams
from sklearn.metrics import f1_score, make_scorer

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

learning_rate = 0.0003

# Initialize Hyperparameters
number_of_classes = 2
embed_dim = 32 # embedding dimension
hidden1 = 128 # (size of the) first hidden layer
hidden2= 16

class TransNet(nn.Module):
    def __init__(self, n_channels=64, n_samples=400, embed_dim=32):
        super(TransNet,self).__init__()
        '''CNN : to capture local features (encodes low-level spatio-temporal information)
         - First layer to get embed_dim filters for detecting features
         - Second layer to extract information along the channels dimension of the EEG data'''
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=(1, 25), stride=(1,1)) #(kernel along time dimension)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, (n_channels, 1), (1,1)) # (kernel along "space" dimension)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)

        # Pooling along the time dimension to smooth/reduce temporal features
        # (prevent overfitting and help computation)
        #self.pool = nn.AvgPool2d((1, 65), (1, 20)) # output shape : (batch, embed_dim, embed_dim, 16)
        self.pool = nn.AvgPool2d((1, 30), (1, 9)) # output shape : (batch, embed_dim, embed_dim, 39)
        self.drop = nn.Dropout(0.5)

        
        # Transformer encoder (attention mechanism for sequencial data)
        self.transformer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        # 2nd Transformer encoder
        #self.transformer2 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)

        # Fully connected layers
        self.fc1 = nn.Linear(embed_dim*39, hidden1) #16
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, number_of_classes)

    def forward(self, obs):
        obs = F.elu(self.norm1(self.conv1(obs)))
        obs = F.elu(self.norm2(self.conv2(obs)))
        obs = self.drop(self.pool(obs))
        # reshape for transformer: expected shape for encoder: (sequence_length, batch_size, embedding_dim)
        obs = torch.reshape(obs,[obs.size(3),obs.size(0), obs.size(1)])
        obs = self.transformer(obs)
        #obs = self.transformer2(obs)

        # flatten for Fully connected
        obs = obs.view(obs.size(1), -1) # flatten along batch_size dimension
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        obs = self.fc3(obs)
        return obs

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.labels = y
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return self.data[id], self.labels[id]

def fit(X_train, y_train, X_val, y_val, iterations=250):

        Xt = torch.FloatTensor(X_train)
        #train_data = Xt.view(Xt.size(0), 1, Xt.size(1), Xt.size(2))
        train_data = Xt.unsqueeze(1)
        print(train_data.size())
        labels = torch.FloatTensor(y_train)

        # DataLoader for batch
        dataset = EEGDataset(train_data, labels)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
        # Keep track of loss at every training iteration
        loss_data = []

        # Begin training for a certain amount of iterations
        for i in range(iterations):
            model.train()
            for k, (X,y) in enumerate(train_loader):

                X,y = X.to(device), y.to(device)
                # Forward pass
                output = model.forward(X)
                #print(output)
                
                # Loss computation
                y_one_hot = torch.nn.functional.one_hot(y.long())
                yf = y_one_hot.float()
                loss = loss_function(output, yf)
                #loss = loss_function(output, y.unsqueeze(1))

                # Zero out the optimizer gradients every iteration
                optimizer.zero_grad()

                # Backpropagation
                loss.backward()
                optimizer.step()

                if k % 2 == 0:
                    print(f"Iteration [{i+1}/{iterations}], Batch [{k+1}/{len(train_loader)}], Loss:{loss.item():.4f}")
                    loss_data.append(loss)

            ## Validation ##
            if i%20 == 0:
                test = torch.FloatTensor(X_val)
                #train_data = Xt.view(Xt.size(0), 1, Xt.size(1), Xt.size(2))
                test_data = test.unsqueeze(1)
                print(test_data.size())
                labs = torch.FloatTensor(y_val)
    
                datatest = EEGDataset(test_data, labs)
                test_loader = DataLoader(datatest, batch_size=64)
                total=0
                correct=0
                model.eval()
                with torch.no_grad():
                    for k, (data,lab) in enumerate(test_loader):
                        data, lab = data.to(device), lab.to(device)
                        out = model(data) #model.forward(Xt)
                        print(out[0:10])
                        _, y_pred = torch.max(out.data,1)
                        total += lab.size(0)
                        correct += (y_pred == lab).sum().item()
                print(f"Accuracy (test): {correct / total:.2f}")

        # Plot loss graph at the end of training
        loss_data_cpu = torch.tensor(loss_data, device = 'cpu') # to convert to numpy
        rcParams['figure.figsize'] = 10, 5
        plt.title("Loss vs Iterations")
        plt.plot(list(range(0, len(loss_data_cpu))), loss_data_cpu)
        plt.show()

def retrieve_data(subject_start,subject_end):
    # retrieve X,y for train or test dataset
    subject_list = [i for i in range(subject_start + 1,subject_end)]
    raw_fnames10 = eegbci.load_data(subject_start, runs)
    all_raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames10])
    for subject in subject_list:
        raw_fnames = eegbci.load_data(subject, runs)
        raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
        all_raw = concatenate_raws([all_raw,raw])

    # standardize
    eegbci.standardize(all_raw)  
    # standard montage
    montage = make_standard_montage("standard_1005")
    all_raw.set_montage(montage)
    
    # Retrieve Events
    events, event_id = mne.events_from_annotations(all_raw)

    tmin, tmax = -0.245, 2.25 
    picks = mne.pick_types(all_raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    epochs = mne.Epochs(all_raw, events, event_id=[2,3], tmin=tmin, tmax=tmax,baseline=None,preload=True,picks=picks)
    #,reject=dict(eeg=150e-6)) # rejection threshold too high
    print(epochs)

    # EEG signals: n_epochs, n_eeg_channels, n_times
    epochs_data = epochs.get_data(copy=False)  
    # labels 0 or 1: hands vs feet
    labels = epochs.events[:, -1] - 2

    ### Preprocessing step ###
    # Apply band-pass filter
    filt = FilterEstimator(epochs.info, 7.0, 30, fir_design='firwin')           
    X = epochs_data
    X = filt.transform(X)
    y = labels
    return X,y

if __name__ == '__main__':
    # Initialize model
    model = TransNet().to(device)

    ## Define a learning function, needs to be reinitialized every load
    learning_rate = 0.0003
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    # Retrieve Data
    runs = [6,10,14]
    X,y = retrieve_data(4,14)
    X_val,y_val = retrieve_data(1,4)

    # TRAIN #
    fit(X,y,X_val,y_val)
