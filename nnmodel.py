import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from pylab import rcParams
import matplotlib as plt
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Initialize parameters
learning_rate = 1e-3 # How hard the network will correct its mistakes while learning
eeg_sample_length = 4 # Number of eeg data points per sample
number_of_classes = 1 #
hidden1 = 10 # Number of neurons in the first hidden layer
hidden2 = 50 # Number of neurons in the second hidden layer
hidden3 = 10 # Number of neurons in the third hidden layer
output1 = 4 # Number of neurons in the output layer

class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.device = device
        ## Define a learning function, needs to be reinitialized every load
        self.optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        self.loss_function = torch.nn.CrossEntropyLoss()
        input_size = eeg_sample_length
        hidden_size = hidden1
        hidden_size2 = hidden2
        hidden_size3 = hidden3
        output_size = output
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, output_size)
        self.output = nn.Linear(output_size, number_of_classes)
    
    def forward(self, obs):
        obs = F.relu(self.fc1(obs))
        obs = F.relu(self.fc2(obs))
        obs = F.relu(self.fc3(obs))
        obs = F.relu(self.fc4(obs))
        obs = F.sigmoid(self.output(obs))
        return obs

    def predict(self, X):
        Xt = torch.FloatTensor(X).to(self.device)
        y_pred = self.forward(Xt)
        y_pred = torch.tensor(y_pred, device = 'cpu').numpy() # convert to numpy
        return np.round(y_pred.squeeze(1))
    
    def fit(self, X_train, y_train, iterations=50):

        train_data = torch.FloatTensor(X_train).to(self.device)
        print(train_data.size())
        labels = torch.FloatTensor(y_train).to(self.device)
        # Keep track of loss at every training iteration
        loss_data = []

        # Begin training for a certain amount of iterations
        for i in range(iterations):

            # Forward pass
            output = self.forward(train_data)
    
            # Loss computation
            loss = self.loss_function(output, labels.unsqueeze(1))
            loss_data.append(loss)

            # Zero out the optimizer gradients every iteration
            self.optimizer.zero_grad()

            # Backpropagation
            loss.backward()
            self.optimizer.step()

        # Plot loss graph at the end of training
        loss_data_cpu = torch.tensor(loss_data, device = 'cpu') # to convert to numpy
        rcParams['figure.figsize'] = 10, 5
        plt.title("Loss vs Iterations")
        plt.plot(list(range(0, len(loss_data_cpu))), loss_data_cpu)
        plt.show()
