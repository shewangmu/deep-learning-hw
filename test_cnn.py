from mlxtend.data import loadlocal_mnist
from cnn import *
from solver import *
import matplotlib.pyplot as plt
import numpy as np

X, y = loadlocal_mnist(
        images_path='./data/raw/train-images-idx3-ubyte', 
        labels_path='./data/raw/train-labels-idx1-ubyte')

x_train = X[:100,:].reshape((-1, 1, 28, 28))
y_train = y[:100]
x_val = X[100:200,:].reshape((-1, 1, 28, 28))
y_val = y[100:200]

x_test, y_test = loadlocal_mnist(
        images_path='./data/raw/t10k-images-idx3-ubyte', 
        labels_path='./data/raw/t10k-labels-idx1-ubyte')
x_test = x_test

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
                      update_rule='sgd',
                      optim_config={
                        'learning_rate': 1e-3,  #hidden layer
                        #'learning_rate': 0.5, #single layer
                      },
                      lr_decay=0.95,
                      num_epochs=50, batch_size=10,
                      print_every=1000)
solver.train()
plt.plot(solver.loss_history)
