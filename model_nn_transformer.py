''' Neural Network with CNN and Transformer encoder'''

import numpy as np
import mne
import pylsl
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.channels import make_standard_montage
from mne.decoding import CSP
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from mne.decoding import FilterEstimator

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

# Initialize Hyperparameters
number_of_classes = 2
embed_dim = 40 #32 # embedding dimension
hidden1 = 256 # (size of the) first hidden layer
hidden2= 32
PATH = "nn_transformer_model2.pt" ## WARNING : Change to your file path ##
num_encoder_layers=1

class TransformerNet(nn.Module):
    def __init__(self, n_channels=64, n_samples=400, embed_dim=40):
        super(TransformerNet,self).__init__()
        self.loss_function = torch.nn.CrossEntropyLoss()
        '''CNN : to capture local features (encodes low-level spatio-temporal information)
         - First layer to get embed_dim filters for detecting features
         - Second layer to extract information along the channels dimension of the EEG data'''
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=(1, 25), stride=(1,1)) #(kernel along time dimension)
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, (n_channels, 1), (1,1)) # (kernel along "space" dimension)
        self.norm1 = nn.BatchNorm2d(embed_dim) #, momentum=0.01)
        self.norm2 = nn.BatchNorm2d(embed_dim) #, momentum=0.01)

        # Pooling along the time dimension to smooth/reduce temporal features
        # (prevent overfitting and help computation)
        #self.pool = nn.AvgPool2d((1, 75), (1, 18)) # output shape : (batch, embed_dim, embed_dim, 40)
        self.pool = nn.AvgPool2d((1, 75), (1, 15)) # output shape : (batch, embed_dim, embed_dim, 47)
        #self.pool = nn.AvgPool2d((1, 65), (1, 12)) # output shape : (batch, embed_dim, embed_dim, 60)
        self.drop = nn.Dropout(0.5)

        
        # Transformer encoder (attention mechanism for sequencial data)
        #self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4)
        encoder_norm = nn.LayerNorm(embed_dim)
        self.encoder = nn.TransformerEncoder(transformer_layer, num_encoder_layers, encoder_norm)

        # Fully connected layers
        self.fc1 = nn.Linear(embed_dim*47, hidden1) 
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.drop3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden2, number_of_classes)

    def forward(self, obs):
        obs = F.elu(self.norm1(self.conv1(obs)))
        obs = F.elu(self.norm2(self.conv2(obs)))
        obs = self.pool(obs)
        #obs = self.drop(obs)
        # reshape for transformer: expected shape for encoder: (sequence_length, batch_size, embedding_dim)
        obs = torch.reshape(obs,[obs.size(3),obs.size(0), obs.size(1)])
        obs = self.encoder(obs)
        #obs = self.transformer_layer(obs)

        # flatten for Fully connected
        obs = obs.view(obs.size(1), -1) # flatten along batch_size dimension
        obs = F.elu(self.fc1(obs))
        #obs = self.drop2(obs)
        obs = F.elu(self.fc2(obs))
        #obs = self.drop3(obs)
        obs = self.fc3(obs)
        return obs

    # get arguments
    def set_optim(self, opti, devi):
        self.optimizer = opti
        self.device = devi

    # Training and validation functions #
    def fit(self, X_train, y_train, X_test, y_test, iterations=10):

        Xt = torch.FloatTensor(X_train)
        train_data = Xt.unsqueeze(1)
        labels = torch.FloatTensor(y_train)

        test = torch.FloatTensor(X_test)
        test_data = test.unsqueeze(1)
        labs = torch.FloatTensor(y_test)

        # DataLoader for batch
        dataset = EEGDataset(train_data, labels)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

        datatest = EEGDataset(test_data, labs)
        test_loader = DataLoader(datatest, batch_size=64, shuffle=False)

        loss_data = [] # to display evolution of train_loss 
    
        # Begin training iterations #
        for i in range(iterations):
            self.train() #model.train()
            total_train=0
            correct_train=0
            for k, (X,y) in enumerate(train_loader):

                X,y = X.to(self.device), y.to(self.device)
                # Forward pass
                output = self.forward(X)
                
                # Loss computation
                y_one_hot = torch.nn.functional.one_hot(y.long(), num_classes=2)
                yf = y_one_hot.float()
                loss = self.loss_function(output, yf)
                loss_data.append(loss)
                _, yt_pred = torch.max(output.data,1)
                total_train += y.size(0)
                correct_train += (yt_pred == y).sum().item()

                # Zero out the optimizer gradients every iteration
                self.optimizer.zero_grad()

                # Backpropagation
                loss.backward()
                self.optimizer.step()

                if k % 2 == 0:
                    print(f"Iteration [{i+1}/{iterations}], Batch [{k+1}/{len(train_loader)}], Loss:{loss.item():.4f}")
                

            print(f"Accuracy (train): {correct_train / total_train:.2f}")

            ## Validation ##
            if i%1 == 0:
                total=0
                correct=0
                #model.eval() # Issue with Batchnorm2d in eval mode
                with torch.no_grad():
                    for k, (data,lab) in enumerate(test_loader):
                        data, lab = data.to(self.device), lab.to(self.device)
                        out = self.forward(data) #model.forward(data)
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
        torch.save(self.state_dict(), PATH)

    def evaluate(self,X_test, y_test):
        test = torch.FloatTensor(X_test)
        test_data = test.unsqueeze(1)
        labs = torch.FloatTensor(y_test)
        datatest = EEGDataset(test_data, labs)
        test_loader = DataLoader(datatest, batch_size=64, shuffle=False)
        total=0
        correct=0
        y_all_true=[]
        y_all_predicted=[]
        
        #self.eval() # Issue with Batchnorm2d in eval mode
        with torch.no_grad():
            for k, (data,lab) in enumerate(test_loader):
                data, lab = data.to(self.device), lab.to(self.device)
                out = self.forward(data)
                _, y_pred = torch.max(out.data,1)
                total += lab.size(0)
                correct += (y_pred == lab).sum().item()
                # f1-score
                y_true = (torch.tensor(lab, device = 'cpu').numpy()).astype(int) # convert to integer scalar array
                y_predicted = (torch.tensor(y_pred, device = 'cpu').numpy()).astype(int) # convert to integer scalar array
                y_all_true = np.concatenate((y_all_true,y_true))
                y_all_predicted = np.concatenate((y_all_predicted,y_predicted))
        print(f"Accuracy NN_Transformer (test): {correct / total:.2f}")
        print(f"F1-score NN_Transformer (test): {f1_score(y_all_true,y_all_predicted, average="weighted"):.2f}")

        
def concat(l1,l2):
    if len(l1) > 0:
        return np.concatenate((l1,l2))
    return l2

## Load Dataset for Training ##

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.labels = y
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, id):
        return self.data[id], self.labels[id]

def retrieve_data(subject_start,subject_end, runs=[6,10,14]):
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

    #tmin, tmax = -0.245, 2.25 
    tmin, tmax = -1, 4
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
    model = TransformerNet().to(device)
    
    ## Define a learning function, needs to be reinitialized every load
    learning_rate = 0.0003
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.set_optim(opti=optimizer, devi=device)

    # Retrieve Data
    runs = [6,10,14]
    X,y = retrieve_data(13,23, runs)
    X_val,y_val = retrieve_data(1,3, runs)

    # TRAIN #
    model.fit(X,y,X_val,y_val)
