'''
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time



class VGG(nn.Module):
    # You will implement a simple version of vgg11 (https://arxiv.org/pdf/1409.1556.pdf)
    # Since the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3, 
    # the number of channels and hidden units are decreased compared to the architecture in paper
    def __init__(self):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            #conv - relu -maxpool -fc -relu -fc -softmax
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
        )
        self.fc = nn.Sequential(
            # TODO: fully-connected layer (64->64)
            # TODO: fully-connected layer (64->10)
            torch.nn.Linear(1568, 100),
            nn.ReLU(),
            torch.nn.Linear(100, 10, bias=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 1568)
        x = self.fc(x)
        return x


def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
#            images = torch.nn.functional.upsample(images, size=(150,150), 
#                                                  scale_factor=None, mode='bilinear', align_corners=None)
            labels = labels.to(device)
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            optimizer.zero_grad()
            outputs = net(images)
            running_loss = criterion(outputs, labels)
            running_loss.backward()
            optimizer.step()
            
            # print statistics
            # running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
                break
    print('Finished Training')


def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def main():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                       download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False)
    net = VGG().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)

    train(trainloader, net, criterion, optimizer, device)
    test(testloader, net, device)
    

if __name__== "__main__":
    main()

    
   
'''

from mlxtend.data import loadlocal_mnist
from cnn import *
from solver import *
import matplotlib.pyplot as plt
import numpy as np

X, y = loadlocal_mnist(
        images_path='./data/raw/train-images-idx3-ubyte', 
        labels_path='./data/raw/train-labels-idx1-ubyte')

x_train = X[:1000,:].reshape((-1, 1, 28, 28))/255
y_train = y[:1000]
x_val = X[1000:2000,:].reshape((-1, 1, 28, 28))/255
y_val = y[1000:2000]

x_test, y_test = loadlocal_mnist(
        images_path='./data/raw/t10k-images-idx3-ubyte', 
        labels_path='./data/raw/t10k-labels-idx1-ubyte')
x_test = x_test[:1000,:].reshape((-1,1,28,28))/255
y_test = y_test[:1000]

data = {
    'X_train': x_train,    # training data
    'y_train': y_train,    # training labels
    'X_val': x_val,        # validation data
    'y_val': y_val         # validation labels
}

#model = SoftmaxClassifier(input_dim=784, hidden_dim=None, weight_scale=1e-2, reg=0)
#learning_rate = 1e-3
#data should be divided by 255

model = ConvNet(num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0)
#no divided by 255

solver = Solver(model, data,
                      update_rule='adam',
                      optim_config={
                        'learning_rate': 1e-2,  #hidden layer
                        #'learning_rate': 0.5, #single layer
                      },
                      lr_decay=0.95,
                      num_epochs=10, batch_size=100,
                      print_every=10)
solver.train()
plt.plot(solver.loss_history)

scores = model.loss(x_test)
y_pred = []
for i in range(len(scores)):
    y_pred.append(np.argmax(scores[i]))
y_pred = np.hstack(y_pred)
acc = np.mean(y_pred==y_test)
print("test accuracy: {}".format(acc))
