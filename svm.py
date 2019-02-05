import numpy as np

from layers import *

class SVM(object):
  """
  A binary SVM classifier with optional hidden layers.
  
  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the model. Weights            #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases (if any) using the keys 'W2' and 'b2'.                        #
    ############################################################################
    self.hidden_dim = hidden_dim
    if hidden_dim is None:
        self.params['W1'] = np.random.normal(scale=weight_scale, size=input_dim)
        self.params['b1'] = np.zeros(1,)
    else:
        self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim,hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(scale=weight_scale, size=hidden_dim)
        self.params['b2'] = np.zeros(1,)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, D)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N,) where scores[i] represents the probability
    that X[i] belongs to the positive class.
    
    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the model, computing the            #
    # scores for X and storing them in the scores variable.                    #
    ############################################################################
    reg = self.reg
    if self.hidden_dim is None:
        w1 = self.params['W1']
        b1 = self.params['b1']
        s_fc, cache = fc_forward(X,w1,b1)
        scores = 1/(1+np.exp(-s_fc))

    else:
        '''
        #relu
        w1 = self.params['W1']
        b1 = self.params['b1']
        w2 = self.params['W2']
        b2 = self.params['b2']
        s_fc1, cache1 = fc_forward(X,w1,b1)
        scores1, s_fc1 = relu_forward(s_fc1)
        s_fc2, cache2 = fc_forward(scores1, w2, b2)
        scores = 1/(1+np.exp(-1*s_fc2))
        
        '''
        #linear
        w1 = self.params['W1']
        b1 = self.params['b1']
        w2 = self.params['W2']
        b2 = self.params['b2']
        s_fc1, cache1 = fc_forward(X,w1,b1)
        s_fc2, cache2 = fc_forward(s_fc1, w2, b2)
        scores = 1/(1+np.exp(-s_fc2))
        
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the model. Store the loss          #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss and make sure that grads[k] holds the gradients for self.params[k]. #
    # Don't forget to add L2 regularization.                                   #
    #                                                                          #
    ############################################################################
    if self.hidden_dim is None:
        loss, ds_fc = svm_loss(s_fc, y)
        loss += reg * np.linalg.norm(w1, ord=2)**2
#        ds_fc = ds * scores * (1-scores)
        dx,dw,db = fc_backward(ds_fc, cache)
        dw = dw + 2*reg*w1
        grads['W1'] = dw
        grads['b1'] = db
    else:
        '''
        #relu
        loss, ds_fc2 = svm_loss(s_fc2, y)
        loss += reg*np.linalg.norm(w2, ord=2)**2
        dscores1, dw2, db2 = fc_backward(ds_fc2, cache2)
        dw2 = dw2 + 2*reg*w2
        ds_fc1 = relu_backward(dscores1, s_fc1)
        dx, dw1, db1 = fc_backward(ds_fc1, cache1)
        grads['W1'] = dw1
        grads['b1'] = db1
        grads['W2'] = dw2
        grads['b2'] = db2
        '''
        #linear
        loss, ds_fc2 = svm_loss(s_fc2, y)
        loss += reg*np.linalg.norm(w2, ord=2)**2
        ds_fc1, dw2, db2 = fc_backward(ds_fc2, cache2)
        dw2 = dw2 + 2*reg*w2
        dx, dw1, db1 = fc_backward(ds_fc1, cache1)
        grads['W1'] = dw1
        grads['b1'] = db1
        grads['W2'] = dw2
        grads['b2'] = db2
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
