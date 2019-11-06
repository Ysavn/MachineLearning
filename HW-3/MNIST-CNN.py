import torch
import torchvision
import datetime
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(20 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)
        x = x.view(-1, 20 * 13 * 13)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def initialize(batchSize):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=True, num_workers=2)
    return (trainloader, testloader)


def training_accuracy(trainloader, net):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct * 100 / total


def test_accuracy(testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %.2f' % (
            100 * correct / total))

def train_neural_network(N, trainloader, net, optimizer, criterion):
    train_loss = []
    train_accuracy = []
    prev_accuracy = 0
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)
    for epoch in range(N):
        running_loss = 0
        for i, data in enumerate(trainloader):
            input, labels = data
            optimizer.zero_grad()
            outputs = net(input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        trn_acc = training_accuracy(trainloader, net)
        print(running_loss, trn_acc)
        if trn_acc <= prev_accuracy:
            break
        prev_accuracy = trn_acc
        train_loss.append(running_loss)
        train_accuracy.append(trn_acc)
    return (train_loss, train_accuracy)
#np.save('train_loss_cnn', np.asarray(train_loss))
#np.save('train_accuracy_cnn', np.asarray(train_accuracy))

def train_neural_network_for_fixed_epoch(N, trainloader, net, optimizer, criterion):
    train_loss = []
    train_accuracy = []
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)
    for epoch in range(N):
        running_loss = 0
        for i, data in enumerate(trainloader):
            input, labels = data
            optimizer.zero_grad()
            outputs = net(input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        trn_acc = training_accuracy(trainloader, net)
        print(running_loss, trn_acc)
        train_loss.append(running_loss)
        train_accuracy.append(trn_acc)
    return (train_loss, train_accuracy)

#train_loss = np.load('train_loss_cnn.npy').tolist()
#train_accuracy = np.load('train_accuracy_cnn.npy').tolist()

def plot_convergence_time_batch_size(batchSize, convergenceTime):
    plt.xlabel('Batch Size')
    plt.ylabel('Convergence Time in minutes')
    plt.plot(batchSize, convergenceTime)
    plt.show()

def cal_covergenceTime_batchSize():
    batchSizes = []
    N = 20
    convergence_time = []
    n_networks = []
    for i, batchSize in enumerate(batchSizes):
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        trainloader, _ = initialize(batchSize)
        time1 = datetime.datetime.now()
        train_neural_network(N , trainloader, net, optimizer, criterion)
        time2 = datetime.datetime.now()
        convergence_time.append((time2-time1).total_seconds()/60)
    #np.save('convergence_time', np.asarray(convergence_time))
    #np.save('batch_size', np.asarray(batchSizes))
    batchSizes = np.load('batch_size.npy').tolist()
    convergence_time = np.load('convergence_time.npy').tolist()
    plot_convergence_time_batch_size(batchSizes, convergence_time)

def plot_loss_for_optimizers_epoch(loss, epoch):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    plt.xlabel('Number of Epochs')
    ax1.set(ylabel = 'Trn Loss SGD')
    ax2.set(ylabel='Trn Loss Adam')
    ax3.set(ylabel='Trn Loss Adagrad')
    ax1.plot(epoch, loss[0], color = 'r')
    ax2.plot(epoch, loss[1], color = 'b')
    ax3.plot(epoch, loss[2], color='g')
    plt.show()


def train_with_different_optimizers():
    net1 = Net()
    net2 = Net()
    net3 = Net()
    N = 8
    trainloader, _ = initialize(32)
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=0.001, momentum=0.9)
    optimizer2 = optim.Adam(net2.parameters(), lr=0.001)
    optimizer3 = optim.Adagrad(net3.parameters(), lr=0.001)
    #loss_SGD,_ = train_neural_network_for_fixed_epoch(N, trainloader, net1, optimizer1, criterion)
    #loss_ADAM,_ = train_neural_network_for_fixed_epoch(N, trainloader, net2, optimizer2, criterion)
    #loss_ADAGRAD,_ = train_neural_network_for_fixed_epoch(N, trainloader, net3, optimizer3, criterion)
    epoch = range(1, 9)
    #np.save('loss_SGD', np.asarray(loss_SGD))
    #np.save('loss_Adam', np.asarray(loss_ADAM))
    #np.save('loss_Adagrad', np.asarray(loss_ADAGRAD))
    loss_SGD = np.load('loss_SGD.npy').tolist()
    loss_ADAM = np.load('loss_Adam.npy').tolist()
    loss_ADAGRAD = np.load('loss_Adagrad.npy').tolist()
    loss = [loss_SGD, loss_ADAM, loss_ADAGRAD]
    plot_loss_for_optimizers_epoch(loss, epoch)






def plot_train_loss_accuracy(N, train_loss, train_accuracy):
    num_epoch = range(1, N+1)
    fig, (ax1, ax2) = plt.subplots(2)
    plt.xlabel('Number of Epochs')
    ax1.set(ylabel='Training Loss')
    ax2.set(ylabel='Training Accuracy')
    ax1.plot(num_epoch, train_loss, color='b')
    ax2.plot(num_epoch, train_accuracy, color='g')
    plt.show()

if __name__ == '__main__':
    trainloader, testloader = initialize(32)
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    #train_neural_network(15, trainloader, net, optimizer, criterion)
    PATH = './mnist_cnn.pth'
    #torch.save(net.state_dict(), PATH)
    #net.load_state_dict(torch.load(PATH))
    #test_accuracy(testloader)
    #cal_covergenceTime_batchSize()
    train_with_different_optimizers()

