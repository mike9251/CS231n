import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    
    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W"""
    
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_samples = X.shape[0] 
    num_labels = W.shape[1]
    for i in range(num_samples):
        scores = X[i].dot(W)            #scores X[i] * W    1 x num_labels
        scores_exp = np.exp(scores)     #exp                1 x num_labels - just a vector with obtained scores for each class label
        norm_scores_exp = scores_exp / np.sum(scores_exp) #norm
        correct_class_score = norm_scores_exp[y[i]] # use correct class label for the current sample (y[i]) to find the score for this class
        loss += -np.log(correct_class_score)
        
        for k in range(num_labels):
            p_k = norm_scores_exp[k]
            dW[:, k] += (p_k - (k == y[i])) * X[i]
                
    loss /= num_samples
    loss += 0.5*reg*np.sum(W*W)
    
    dW /= num_samples
    dW += reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """  Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    
    loss = 0.0
    dW = np.zeros_like(W)
    
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_samples = X.shape[0] 
    
    scores = X.dot(W) # num_samples x num_labels
        
    scores_exp = np.exp(scores) # matrix num_samples x num_labels, contains scores of each class label for the entire X
    
    norm_scores_exp = scores_exp / np.sum(scores_exp, axis = 1, keepdims=True)
    
    correct_class_score = norm_scores_exp[np.arange(num_samples), y]# a vector num_samples x 1, contains scores of correct classes label 
    
    loss = np.sum(-np.log(correct_class_score))
    
    ind = np.zeros_like(norm_scores_exp)#mask with 0 norm_scores_exp.shape
    ind[np.arange(num_samples), y] = 1  #the indecies for correct classes = 1
    dW = X.T.dot(norm_scores_exp - ind) #so in () we satisfy an condition that if k==y[i] then norm_scores_exp - 1. k - current class label
                
    loss /= num_samples
    loss += 0.5*reg*np.sum(W*W)
    
    dW /= num_samples
    dW += reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW

