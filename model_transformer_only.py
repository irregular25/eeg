''' Neural Network with only Transformer encoder
This model is given for reference but it doesn't seem to converge with this architecture.
This shows the benefits of adding a CNN layer for dimension reduction and capturing local spatio-temporal features'''

import numpy as np
import mne
import pylsl
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.decoding import CSP
import matplotlib.pyplot as plt
import numpy as np
import math

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
embed_dim = 64 #40 #32 # embedding dimension
hidden1 = 512 #256 # (size of the) first hidden layer
hidden2= 32
num_encoder_layers = 3

# Classical positional encoding #
class PositionalEncoding(nn.Module):
    """Add information about the relative or absolute position of the samples
        in the sequence (positional encodings have the same dimension as
        the embeddings, so that the two can be summed). 
        Here, sine and cosine functions of different frequencies are used.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Args: x is the sequence to be fed to the encoder.
        Shape: x: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerOnlyNet(nn.Module):
    def __init__(self, n_channels=64, n_samples=400, embed_dim=64):
        super(TransformerOnlyNet,self).__init__()
        ''' No dimension reduction here, this model aims to test the transformer encoder performance on EEG data
        without much processing (just pass-band filter) '''
        # Positional Encoding #
        self.positional_encoder = PositionalEncoding(d_model=embed_dim)
        # Transformer Encoder (attention mechanism to capture context in sequencial data) #
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        encoder_norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.TransformerEncoder(transformer_layer, num_encoder_layers, encoder_norm)

        # Fully connected layers on top to get classification #
        self.fc1 = nn.Linear(embed_dim*400, hidden1) #16 # 47
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.drop3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden2, number_of_classes)

    def forward(self, obs):

        # reshape for transformer: expected shape for encoder: (sequence_length, batch_size, embedding_dim)
        obs = torch.reshape(obs,[obs.size(2),obs.size(0), obs.size(1)])
        # Positional Encoder
        obs = self.positional_encoder(obs)
        # Encoder layer
        obs = self.encoder(obs)

        # flatten for Fully connected
        obs = obs.view(obs.size(1), -1) # flatten along batch_size dimension
        obs = F.elu(self.fc1(obs))
        obs = self.drop2(obs)
        obs = F.elu(self.fc2(obs))
        obs = self.drop3(obs)
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

def fit(X_train, y_train, X_test, y_test, iterations=30):

        Xt = torch.FloatTensor(X_train)
        train_data = Xt
        #train_data = Xt.view(Xt.size(0), 1, Xt.size(1), Xt.size(2))
        #print(train_data.size())
        labels = torch.FloatTensor(y_train)

        test = torch.FloatTensor(X_test)
        test_data = test
        #print(test_data.size())
        labs = torch.FloatTensor(y_test)

        # DataLoader for batch
        dataset = EEGDataset(train_data, labels)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        datatest = EEGDataset(test_data, labs)
        test_loader = DataLoader(datatest, batch_size=64, shuffle=False)

        loss_data=[] # keep track of training loss
    
        # Begin training iterations #
        for i in range(iterations):
            model.train()
            total_train=0
            correct_train=0
            for k, (X,y) in enumerate(train_loader):

                X,y = X.to(device), y.to(device)
                # Forward pass
                output = model.forward(X)
                #print(output)
                
                # Loss computation
                y_one_hot = torch.nn.functional.one_hot(y.long(), num_classes=2)
                yf = y_one_hot.float()
                loss = loss_function(output, yf)
                #loss = loss_function(output, y.unsqueeze(1))
                _, yt_pred = torch.max(output.data,1)
                total_train += y.size(0)
                correct_train += (yt_pred == y).sum().item()

                # Zero out the optimizer gradients every iteration
                optimizer.zero_grad()

                # Backpropagation
                loss.backward()
                optimizer.step()

                if k % 2 == 0:
                    print(f"Iteration [{i+1}/{iterations}], Batch [{k+1}/{len(train_loader)}], Loss:{loss.item():.4f}")
                    loss_data.append(loss)

            print(f"Accuracy (train): {correct_train / total_train:.2f}")

            ## Validation ##
            if i%1 == 0:
                total=0
                correct=0
                model.eval()
                with torch.no_grad():
                    for k, (data,lab) in enumerate(test_loader):
                        data, lab = data.to(device), lab.to(device)
                        out = model.forward(data)
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
    #tmin, tmax = -1, 4
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
    model = TransformerOnlyNet().to(device)

    ## Define a learning function, needs to be reinitialized every load
    learning_rate = 0.0003
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()

    # Retrieve Data
    runs = [6,10,14]
    X,y = retrieve_data(8,87)
    X_val,y_val = retrieve_data(1,8)

    # TRAIN #
    fit(X,y,X_val,y_val)
