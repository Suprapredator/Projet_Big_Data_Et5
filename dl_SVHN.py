import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
import time
torch.manual_seed(0)

def numberGoodInTensors(testTensor, labelTensor):
    result = 0
    for i in range(len(testTensor)):
        if testTensor[i] == labelTensor[i]:
            result += 1
    return result

class CNN(nn.Module):

    def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 20, 5)
            self.conv2 = nn.Conv2d(20, 50, 5)
            self.fc1 = nn.Linear(1250, 500)
            self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2(x), 2))
            x = x.view(x.shape[0], -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.log_softmax(self.fc2(x), dim=1)

            return x

class LeNet(nn.Module):

    def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(400, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
            x = F.relu(F.max_pool2d(self.conv1(x), (2,2)))
            x = F.relu(F.max_pool2d(self.conv2(x), (2,2)))
            x = x.view(x.shape[0], -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.log_softmax(self.fc3(x), dim=1)

            return x

class MLP(nn.Module):

    def __init__(self):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(3072, 120)
            self.fc2 = nn.Linear(120, 800)
            self.fc3 = nn.Linear(800, 1250)
            self.fc4 = nn.Linear(1250, 500)
            self.fc5 = nn.Linear(500, 120)
            self.fc6 = nn.Linear(120, 10)


    def forward(self, x):
            x = x.contiguous()
            x = x.view(x.shape[0], -1) # Flatten the tensor
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = F.log_softmax(self.fc6(x), dim=1)

            return x

def testReseau(train_data, train_label, test_data, test_label, epoch_nbr, batch_size, learning_rate, name):
    # Variables    
    bonneReponses = 0
    start_time = time.time()

    if name == "LeNet":
        print("\n*** LeNet ***")
        net = LeNet()
    if name == "CNN":
        print("\n*** CNN ***")
        net = CNN()
    if name == "MLP":
        print("\n*** MLP ***")
        net = MLP()
        
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    for e in range(epoch_nbr):
        bonneReponses = 0
        for i in range(0, train_data.shape[0], batch_size):
            optimizer.zero_grad() # Reset all gradients to 0
            predictions_train = net(train_data[i:i+batch_size])
            _, class_predicted = torch.max(predictions_train, 1)
            
            #print(class_predicted)
            #print(train_label[i:i+batch_size])
            #print(numberGoodInTensors(class_predicted, train_label[i:i+batch_size]))
            
            bonneReponses += numberGoodInTensors(class_predicted, train_label[i:i+batch_size])
            loss = F.nll_loss(predictions_train, train_label[i:i+batch_size])
            loss.backward()
            optimizer.step() # Perform the weights update
        print("Epoch "+str(e)+": "+str(bonneReponses*100/train_data.shape[0])+"% ("+str(bonneReponses)+"/"+str(train_data.shape[0])+")")
    
    print("\n* Essais sur l'ensemble des données:")
    predictions_train = net(train_data)
    _, class_predicted = torch.max(predictions_train, 1)
    bonneReponses = numberGoodInTensors(class_predicted, train_label)
    print("Train = "+str(bonneReponses*100/train_data.shape[0])+"% ("+str(bonneReponses)+"/"+str(train_data.shape[0])+")")
    
    predictions_train = net(test_data)
    _, class_predicted = torch.max(predictions_train, 1)
    bonneReponses = numberGoodInTensors(class_predicted, test_label)
    print("Test = "+str(bonneReponses*100/test_data.shape[0])+"% ("+str(bonneReponses)+"/"+str(test_data.shape[0])+")")
    
    print("\n---"+str(time.time()-start_time)+" sec ---")

if __name__ == '__main__':

    # Load the dataset
    train_data = loadmat('../perfect_train_data.mat')
    test_data = loadmat('../perfect_test_data.mat')

    train_label = train_data['y']
    train_label = np.where(train_label==10, 0, train_label)
    train_label = torch.from_numpy(train_label.astype('int')).squeeze(1)
    train_data = torch.from_numpy(train_data['X'].astype('float32')).permute(3, 2, 0, 1)

    test_label = test_data['y']
    test_label = np.where(test_label==10, 0, test_label)
    test_label = torch.from_numpy(test_label.astype('int')).squeeze(1)
    test_data = torch.from_numpy(test_data['X'].astype('float32')).permute(3, 2, 0, 1)

    # Hyperparameters
    epoch_nbr = 10
    batch_size = 10
    learning_rate = 1e-3
    
    #testReseau(train_data, train_label, test_data, test_label, epoch_nbr, batch_size, learning_rate, "LeNet")
    testReseau(train_data, train_label, test_data, test_label, epoch_nbr, batch_size, learning_rate, "CNN")
    #testReseau(train_data, train_label, test_data, test_label, epoch_nbr, batch_size, learning_rate, "MLP")