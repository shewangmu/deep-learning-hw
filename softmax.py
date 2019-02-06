import numpy as np
from layers import *


class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights                                  #
        # and biases using the keys 'W' and 'b'                                    #
        ############################################################################
        self.hidden_dim = hidden_dim
        if hidden_dim is None or hidden_dim==0:
            self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, num_classes))
            self.params['b1'] = np.zeros(num_classes,)
        else:
            self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim,hidden_dim))
            self.params['b1'] = np.zeros(hidden_dim)
            self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim,num_classes))
            self.params['b2'] = np.zeros(num_classes,)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        hidden_dim = self.hidden_dim
        if hidden_dim == None or hidden_dim==0:
            w = self.params['W1']
            b = self.params['b1']
            s_fc, cache = fc_forward(X,w,b)
            exp = np.exp(s_fc)
            scores = [i/np.sum(i) for i in exp]
            scores = np.array(scores)
        else:
            w1 = self.params['W1']
            b1 = self.params['b1']
            w2 = self.params['W2']
            b2 = self.params['b2']
            s_fc1, cache1 = fc_forward(X, w1, b1)
            s_relu, s_fc1 = relu_forward(s_fc1)
            s_fc2, cache2 = fc_forward(s_relu, w2, b2)
            exp = np.exp(s_fc2)
            scores = [i/np.sum(i) for i in exp]
            scores = np.array(scores)
            '''
            s_fc1, cache1 = fc_forward(X, w1, b1)
            s_fc2, cache2 = fc_forward(s_fc1, w2, b2)
            exp = np.exp(s_fc2)
            scores = [i/np.sum(i) for i in exp]
            scores = np.array(scores)
            '''
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        reg = self.reg
        loss, dscores = softmax_loss(scores, y)
        dexp = np.zeros_like(dscores)
        for i in range(len(dscores)):
            l = y[i]
            dexp[i] = dscores[i]*(np.sum(exp[i])-exp[i][l])/np.sum(exp[i])**2
        if hidden_dim == None or hidden_dim==0:
            ds_fc = dexp * exp
            loss += reg*np.linalg.norm(w, ord=2)**2
            dx, dw, db = fc_backward(ds_fc, cache)
            dw += 2*reg*w
            grads['W1'] = dw
            grads['b1'] = db
        else:

            ds_fc2 = dexp * exp
            loss += reg*np.linalg.norm(w2, ord=2)**2
            ds_relu, dw2, db2 = fc_backward(ds_fc2, cache2)
            ds_fc1 = relu_backward(ds_relu, s_fc1)
            dx, dw1, db1 = fc_backward(ds_fc1, cache1)
            grads['W1'] = dw1
            grads['b1'] = db1
            grads['W2'] = dw2
            grads['b2'] = db2
            '''
            ds_fc2 = dexp * exp
            loss += reg*np.linalg.norm(w2, ord=2)**2
            ds_fc1, dw2, db2 = fc_backward(ds_fc2, cache2)
            dx, dw1, db1 = fc_backward(ds_fc1, cache1)
            grads['W1'] = dw1
            grads['b1'] = db1
            grads['W2'] = dw2
            grads['b2'] = db2
            '''
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
