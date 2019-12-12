import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    nums = X.shape[0]
    y_1_hot = np.zeros((X.shape[0],W.shape[1]))
    for i in range(0,X.shape[0]):
        y_1_hot[i,y[i]] = 1
    y_pred = np.dot(X,W)
    y_soft = np.zeros_like(y_1_hot)
    y_grad = np.zeros_like(y_1_hot)
    for i in range(0,y_pred.shape[0]):
        maximum = np.amax(y_pred[i])
        y_soft[i] = (y_pred[i] - maximum)
        y_soft[i] = np.exp(y_soft[i])
        summa = np.sum(y_soft[i])
        y_soft[i] = y_soft[i]/summa
        y_grad[i] = y_soft[i] # To be used later
        y_soft[i] = np.log(y_soft[i])
        loss_new = np.sum(y_1_hot[i]*y_soft[i])
        loss = loss + loss_new
    loss = -1*loss/y_pred.shape[0]
    loss = loss + reg*np.sum(np.square(W))/2
    for i in range(0,X.shape[0]):
        y_grad[i,y[i]] -= 1
    X_transpose = np.transpose(X)
    y_grad_transpose = np.transpose(y_grad)
    for i in range(0,X_transpose.shape[0]):
        for j in range(0, y_grad_transpose.shape[0]):
            dW[i,j] = np.dot(X_transpose[i,:],y_grad_transpose[j,:])
    dW /= nums
    dW +=reg*W
        
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################

    
    y_1_hot = np.eye(W.shape[1])[y]
    nums = X.shape[0]
    y_pred = np.dot(X,W)
    y_max  = np.amax(y_pred, axis = 1)
    exps = np.exp(y_pred - (np.amax(y_pred,axis=1)).reshape((y_pred.shape[0],1)))
    y_soft_base = exps/(np.sum(exps,axis=1).reshape((y_pred.shape[0],1)))
    y_soft_only_max = (np.amax(y_pred,axis=1)).reshape((y_pred.shape[0],1))/(np.sum(exps,axis=1).reshape((y_pred.shape[0],1)))
    y_soft = np.log(y_soft_base)
    loss = -1*np.sum(y_1_hot*y_soft)/y_pred.shape[0]
    loss  = loss + reg*np.sum(np.square(W))/2
    y_soft_base[range(nums),y] -= 1
    dW = np.dot(np.transpose(X),y_soft_base)
    dW /= nums
    dW += reg*W
    
    #dW = -1*np.dot(np.transpose(X),(y_soft_base - y_soft_only_max))/y_pred.shape[0] + (2*reg*np.sum(W))
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW
