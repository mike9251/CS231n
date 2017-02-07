import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
    
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input channels."""
    
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """Initialize a new network.
        
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
        self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale * np.random.randn(num_filters * (input_dim[1] // 2) * (input_dim[2] // 2), hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes) 

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        for k, v in iter(self.params.items()):
            self.params[k] = v.astype(dtype)
            
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet in fc_net.py."""
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        conv_out_rows = conv_out.reshape(conv_out.shape[0], -1)
        affine_out, affine_cache = affine_relu_forward(conv_out_rows, W2, b2)
        scores, affine2_cache = affine_forward(affine_out, W3, b3)
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
        loss, dout = softmax_loss(scores, y)
        loss += self.reg * 0.5 * ((W1 ** 2).sum() + (W2 ** 2).sum() + (W3 ** 2).sum())
        dx, dW3, db3 = affine_backward(dout, affine2_cache)
        dx, dW2, db2 = affine_relu_backward(dx, affine_cache)
        dx = dx.reshape(conv_out.shape)
        dx, dW1, db1 = conv_relu_pool_backward(dx, conv_cache)
        
        dW3 += self.reg * np.sum(W3)
        dW2 += self.reg * np.sum(W2)
        dW1 += self.reg * np.sum(W1)
        
        grads.update({
                'W1': dW1,
                'b1': db1,
                'W2': dW2,
                'b2': db2,
                'W3': dW3,
                'b3': db3})
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
    
class ConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    
    layer_defs.push({type:'input', out_sx:32, out_sy:32, out_depth:3});
    layer_defs.push({type:'conv', sx:5, filters:16, stride:1, pad:2, activation:'relu'});
    layer_defs.push({type:'pool', sx:2, stride:2});
    layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
    layer_defs.push({type:'pool', sx:2, stride:2});
    layer_defs.push({type:'conv', sx:5, filters:20, stride:1, pad:2, activation:'relu'});
    layer_defs.push({type:'pool', sx:2, stride:2});
    layer_defs.push({type:'softmax', num_classes:10});
    
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input channels."""
    
    def __init__(self, input_dim=(3, 32, 32), num_filters=[16, 20, 20], filter_size=[5, 5, 5],
                 hidden_dim=10, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """Initialize a new network.
        
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
        self.params['W1'] = weight_scale * np.random.randn(num_filters[0], input_dim[0], filter_size[0], filter_size[0])
        self.params['b1'] = np.zeros(num_filters[0])
        
        #print('W1: ', self.params['W1'].shape)
        width = input_dim[1] // 2
        
        self.params['W2'] = weight_scale * np.random.randn(num_filters[1], num_filters[0], filter_size[1], filter_size[1])
        self.params['b2'] = np.zeros(num_filters[1])
        
        width = width // 2
        #print('W2: ', self.params['W2'].shape)
        
        self.params['W3'] = weight_scale * np.random.randn(num_filters[2], num_filters[1], filter_size[2], filter_size[2])
        self.params['b3'] = np.zeros(num_filters[2])
        
        width = width // 2
        #print('W3: ', self.params['W3'].shape)
        
        width = num_filters[2] * width**2
        self.params['W4'] = weight_scale * np.random.randn(width, num_classes)
        self.params['b4'] = np.zeros(10) 
        
        #print('W4: ', self.params['W4'].shape)
        #self.params['W4'] = weight_scale * np.random.randn(10, num_filters[2], 4, 4)
        #self.params['b4'] = np.zeros(10) 
        #self.params['W5'] = weight_scale * np.random.randn(10, num_classes)
        #self.params['b5'] = np.zeros(num_classes) 

        ############################################################################
        #                             END OF YOUR CODE                             #
        ####################################### #####################################
        
        for k, v in iter(self.params.items()):
            self.params[k] = v.astype(dtype)
            
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet in fc_net.py."""
        
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        #W5, b5 = self.params['W5'], self.params['b5']
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = [W1.shape[2], W2.shape[2], W3.shape[2], W4.shape[0]]#, W5.shape[2]]
        
        conv_param = {'stride': 1, 'pad': (filter_size[0] - 1) // 2}
        
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        
        #print('conv_out: ', conv_out.shape)
        
        conv_out2, conv_cache2 = conv_relu_pool_forward(conv_out, W2, b2, conv_param, pool_param)
        
        #print('conv_out2: ', conv_out2.shape)
        
        conv_out3, conv_cache3 = conv_relu_pool_forward(conv_out2, W3, b3, conv_param, pool_param)
        
        #print('conv_out3: ', conv_out3.shape)
        
        conv_out_rows = conv_out3.reshape(conv_out3.shape[0], -1)
        
        #print('conv_out_rows: ', conv_out_rows.shape)
        
        #affine_out, affine_cache = affine_relu_forward(conv_out_rows, W2, b2)
        scores, affine2_cache = affine_forward(conv_out_rows, W4, b4)
        
        #print('scores: ', scores.shape)
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
        loss, dout = softmax_loss(scores, y)

        loss += self.reg * 0.5 * ((W1 ** 2).sum() + (W2 ** 2).sum() + (W3 ** 2).sum() + (W4 ** 2).sum())
        dx, dW4, db4 = affine_backward(dout, affine2_cache)
        
        dx = dx.reshape(conv_out3.shape)
        
        dx, dW3, db3 = conv_relu_pool_backward(dx, conv_cache3)
        
        dx, dW2, db2 = conv_relu_pool_backward(dx, conv_cache2)
        
        dx, dW1, db1 = conv_relu_pool_backward(dx, conv_cache)
        
        dW4 += self.reg * np.sum(W4)
        dW3 += self.reg * np.sum(W3)
        dW2 += self.reg * np.sum(W2)
        dW1 += self.reg * np.sum(W1)
        
        grads.update({
                'W1': dW1,
                'b1': db1,
                'W2': dW2,
                'b2': db2,
                'W3': dW3,
                'b3': db3,
                'W4': dW4,
                'b4': db4})
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads