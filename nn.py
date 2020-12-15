import numpy as np
import torch
import torch.utils.data as utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def convertVoteToDigit(c):
    if c == 'republican':
        return 1
    elif c == 'democrat':
        return 0
    elif c == 'y':
        return 1
    elif c == 'n':
        return -1
    else:
        return 0


def getData(file):
    votes = []
    lines = open(file, "r").readlines()
    for line in lines:
        line = line.replace('\n', '').split(',')
        for l in line:
            votes.append(convertVoteToDigit(l))
    votes = np.reshape(votes, (-1, 17))
    target = votes[:, 0]
    data = votes[:, 1:]
    return data, target

def getTrainTestData(data, target, nb_train_data):
    train_data, train_target = data[0:nb_train_data,:], target[0:nb_train_data]
    test_data, test_target = data[nb_train_data:,:], target[nb_train_data:]
    return train_data, train_target, test_data, test_target


class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 16
num_classes = 2
learning_rate = 0.001
batch_size = 8
num_epochs = 1

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

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape
        #data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(data.float())
        print(scores.shape)
        break
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        print(loss)


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