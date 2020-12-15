import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as utils
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F
import random
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# one_hot_vector = [y,?,n]
def convertOutputToDigit(c):
    if c == 'republican':
        return 1
    else:
        return 0

def convertInputToOneHotVector(c):
    if c == 'y':
        return [1, 0, 0]
    elif c == '?':
        return [0, 1, 0]
    else:
        return [0, 0, 1]

def getData(file):
    outputs = []
    inputs = []
    lines = open(file, "r").readlines()
    for line in lines:
        line = line.replace('\n', '').split(',')
        for l in line:
            if len(l) > 1:
                outputs.append(convertOutputToDigit(l))
            else:
                inputs.append(convertInputToOneHotVector(l))

    inputs = np.reshape(inputs, (-1, 16, 3))
    return inputs, outputs


def getTrainTestData(data, target, nb_train_data):
    train_data, train_target = data[0:nb_train_data,:], target[0:nb_train_data]
    test_data, test_target = data[nb_train_data:,:], target[nb_train_data:]
    return train_data, train_target, test_data, test_target


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 3
hidden_size = 128
num_layers = 2
num_classes = 2
sequence_length = 16
learning_rate = 0.005
batch_size = 8
num_epochs = 2


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


data, target = getData("data.txt")
train_data, train_target, test_data, test_target = getTrainTestData(data, target, 335)

train_data_tensor = torch.tensor(train_data)
train_target_tensor = torch.tensor(train_target)

test_data_tensor = torch.tensor(test_data)
test_target_tensor = torch.tensor(test_target)

train_dataset = utils.TensorDataset(train_data_tensor,train_target_tensor)
test_dataset = utils.TensorDataset(test_data_tensor,test_target_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# print(train_dataset[0])
# print(next(iter(train_loader)))

# Initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data.float())

        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        print(loss)
#
#
def check_accuracy(loader, model):

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x.float())
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )


#check_accuracy(train_loader, model)
check_accuracy(test_loader, model)