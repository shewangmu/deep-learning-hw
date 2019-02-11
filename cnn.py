import numpy as np

from layers import *


class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - relu - fc -softmax

  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    F, HH, WW = num_filters, filter_size, filter_size
    self.params['W1'] = np.random.normal(scale = weight_scale, size = (F, C, HH, WW))
    self.params['b1'] = np.zeros((F, ))
    self.params['W2'] = np.random.normal(scale = weight_scale, size = (int(F*H*W/4), hidden_dim))
    self.params['b2'] = np.zeros((hidden_dim,))
    self.params['W3'] = np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes))
    self.params['b3'] = np.zeros((num_classes,))
    self.params['gamma'] = np.random.normal(scale = weight_scale, size = hidden_dim)
    self.params['beta'] = np.random.normal(scale = weight_scale, size = hidden_dim)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    
    pad_num = int(conv_param['pad'])
    x_pad = np.pad(X, ((0,0),(0,0),(pad_num,pad_num),(pad_num,pad_num)), 'constant', constant_values=(0,0))
    
    s_conv, cache1 = conv_forward(x_pad, W1)
    s_conv_relu, s_conv = relu_forward(s_conv)
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    s_max, cache2 = max_pool_forward(s_conv_relu, pool_param)
    
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    s_max_size = s_max.shape
    s_max = s_max.reshape((s_max_size[0],-1))
    gamma = self.params['gamma']
    beta = self.params['beta']
    if y is not None:
        bn_param = {'mode':'train'}
        dropout_param = {'mode':'train', 'p':0.7}
    else:
        bn_param = {'mode':'test'}
        dropout_param = {'mode':'test','p':0.7}
    
    s_fc1, cache3 = fc_forward(s_max, W2, b2)
    s_norm, cache_norm = batchnorm_forward(s_fc1, gamma, beta, bn_param)
    s_dropout, cache_drop = dropout_forward(s_norm, dropout_param)
    
    s_relu1, s_fc1 = relu_forward(s_fc1)
    s_fc2, cache4 = fc_forward(s_relu1, W3, b3)
    
    s_fc2[s_fc2>13] = 13
    exp = np.exp(s_fc2)
    scores = [i/np.sum(i) for i in exp]
    scores = np.array(scores)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, ds_fc2 = softmax_loss(s_fc2, y)
    ds_relu1, dw3, db3 = fc_backward(ds_fc2, cache4)
    ds_fc1 = relu_backward(ds_relu1, s_fc1)
    
    ds_norm = dropout_backward(ds_fc1, cache_drop)
    ds_fc1, dgamma, dbeta = batchnorm_backward(ds_norm, cache_norm)
    
    ds_max, dw2, db2 = fc_backward(ds_fc1, cache3)    
    ds_max = ds_max.reshape(s_max_size)
    ds_conv_relu = max_pool_backward(ds_max, cache2)
    ds_conv = relu_backward(ds_conv_relu, s_conv)
    dx, dw1 = conv_backward(ds_conv, cache1)
    db1 = np.zeros_like(b1)
    
    grads['W1'] = dw1
    grads['b1'] = db1
    grads['W2'] = dw2
    grads['b2'] = db2
    grads['W3'] = dw3
    grads['b3'] = db3
    grads['gamma'] = dgamma
    grads['beta'] = dbeta
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

