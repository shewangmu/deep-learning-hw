import numpy as np

from layers import *

class LogisticClassifier(object):
  """
  A logistic regression model with optional hidden layers.
  
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
        self.params['w1'] = np.random.normal(scale=weight_scale, size=input_dim)
        self.params['b1'] = np.zeros(1,)
    
    else:
        self.params['w1'] = np.random.normal(scale=weight_scale, size=(input_dim,hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim,)
        self.params['w2'] = np.random.normal(scale=weight_scale, size=hidden_dim)
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
    if self.hidden_dim is None:
        w1 = self.params['w1']
        b1 = self.params['b1']
        y_pred, cache = fc_forward(X, w1, b1)
        scores = 1/(np.exp(-y_pred)+1)
    else:
        w1 = self.params['w1']
        b1 = self.params['b1']
        y_pred1, cache = fc_forward(X, w1, b1)
        scores1 = 1/(np.exp(-y_pred1)+1)   #sigmoid
        w2 = self.params['w2']
        b2 = self.params['b2']
        y_pred2, cache2 = fc_forward(scores1, w2, b2)
        scores = 1/(np.exp(-y_pred2)+1)
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
    reg = self.reg
    if self.hidden_dim is None:
        loss, dout = logistic_loss(y_pred, y)
        loss += reg * np.linalg.norm(w1, ord=2)**2
        dx, dw1, db1 = fc_backward(dout, cache)
        grads['w1'] = dw1 + reg*2*w1
        grads['b1'] = db1 
    else:
        loss, dout2 = logistic_loss(y_pred2, y)
        loss += reg * np.linalg.norm(w2, ord=2)**2
        dout1, dw2, db2 = fc_backward(dout2, cache2)
        grads['w2'] = dw2 + reg*2*w2
        grads['b2'] = db2
        dout = dout1 * scores1 * (1-scores1)
        dx, dw1, db1 = fc_backward(dout, cache)
        grads['w1'] = dw1 + reg*2*w1
        grads['b1'] = db1
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
