import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(root = './data', train=True, download=True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size =32, shuffle=True, num_workers = 2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download = True, transform = transform)
testloader = torch.utils.data.DataLoader(testset, batch_size =32, shuffle=True, num_workers = 2)

#Fully Connected Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x, dim=1)
        return x

#calculate Training accuracy
def training_accuracy(trainloader):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct*100/total

#calculate Test Accuracy
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

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

#training the model for 15 epochs, currently N = 0 as using pretrained model saved as mnist-fc.pth
N = 0
train_loss = []
train_accuracy = []
for epoch in range(N):
    running_loss = 0
    for i, data in enumerate(trainloader):
        input, labels = data
        optimizer.zero_grad()
        outputs = net(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.shape[0]
    running_loss /= len(trainset)
    #print(running_loss, training_accuracy(trainloader))
    train_loss.append(running_loss)
    train_accuracy.append(training_accuracy(trainloader))

#np.save('train_loss', np.asarray(train_loss))
#np.save('train_accuracy', np.asarray(train_accuracy))

PATH = './mnist_fc.pth'
#torch.save(net.state_dict(), PATH)

net.load_state_dict(torch.load(PATH))
train_loss = np.load('train_loss.npy').tolist()
train_accuracy = np.load('train_accuracy.npy').tolist()

num_epoch = range(1, 16)
fig, (ax1, ax2) = plt.subplots(2)

plt.xlabel('Number of Epochs')
ax1.set(ylabel = 'Training Loss')
ax2.set(ylabel='Training Accuracy')
ax1.plot(num_epoch, train_loss, color = 'b')
ax2.plot(num_epoch, train_accuracy, color = 'g')
plt.show()

#Accuracy on the test set
test_accuracy(testloader)
