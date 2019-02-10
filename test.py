#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:02:33 2019

@author: shengwang
"""
import matplotlib.pyplot as plt

from logistic import *
from solver import *
from svm import *
import numpy as np
import pickle
with open('data.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

X, Y = data
x_train = X[:500,:]
y_train = Y[:500]
x_val = X[500:750,:]
y_val = Y[500:750]
x_test = X[750:,:]
y_test = Y[750:]

data = {
    'X_train': x_train,    # training data
    'y_train': y_train,    # training labels
    'X_val': x_val,        # validation data
    'y_val': y_val         # validation labels
}

model_1 = LogisticClassifier(input_dim=20, hidden_dim=30, weight_scale=1, reg=0)
#model_2 = SVM(input_dim=20, hidden_dim=30, weight_scale=1, reg=0)

def train(model, learning_rate=5e-2):
    solver = Solver(model, data,
                      update_rule='sgd',
                      optim_config={
                        'learning_rate': learning_rate,  #hidden layer
                        #'learning_rate': 0.5, #single layer
                      },
                      lr_decay=0.95,
                      num_epochs=50, batch_size=10,
                      print_every=100)
    solver.train()
    plt.plot(solver.loss_history)
    
    scores = model.loss(x_test)
    y_pred = [int(round(i)) for i in scores]
    acc = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            acc +=1
    print("test accuracy is:{}".format(acc/250))
    return y_pred
    
y_pred_svm = train(model_2, learning_rate = 1e-2)
print("SVM")
y_pred_logistic = train(model_1, learning_rate=5e-1)
print("Logistic")
