import numpy as np
from random import shuffle

def sigmoid(x):
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                            #         
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    new = np.exp(x)/(1+ np.exp(x))
    h = new
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################
    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 2.3
    # Initialize the gradient to zero
    dW = 0

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    y_pred = np.dot(X,W)
    sig_y_pred = sigmoid(y_pred)
    L = 0
    y_new = np.zeros((y.shape[0], W.shape[1]))
    y_1 = np.asarray([[1-x,x] for x in y])
    for i in range (0,sig_y_pred.shape[0]):
        for j in range(0,sig_y_pred.shape[1]):
            L = L + (y_1[i,j]*np.log(sig_y_pred[i][j])) + ((1-y_1[i,j])*np.log(1-sig_y_pred[i][j])) 
        if y[i] == 1:
            y_new[i][1] = 1
        else:
            y_new[i][0] = 1
    
        
    L = -1*(L/sig_y_pred.shape[0])
    loss = L + 0.5*reg*(np.sum(np.square(W)))
    dW = np.dot(np.transpose(X), (sig_y_pred- y_new))/y_new.shape[0] + reg*W
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 3.5
    # Initialize the gradient to zero
    dW = 0

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no    # 
    # explicit loops.                                       #
    # Store the loss in loss and the gradient in dW. If you are not careful   #
    # here, it is easy to run into numeric instability. Don't forget the     #
    # regularization!                                       #
    ############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    y_pred = np.dot(X,W)
    sig_y_pred = sigmoid(y_pred)
    y_new = np.zeros((y.shape[0], W.shape[1]))
    y_new =np.asarray([[1-x,x] for x in y])
    L = np.multiply(y_new,np.log(sig_y_pred)) + np.multiply((np.ones_like(y_new)-y_new),np.log(np.ones_like(sig_y_pred)-sig_y_pred))
    L = -1*np.sum(L)/sig_y_pred.shape[0]
    loss = L + 0.5*reg*(np.sum(np.square(W)))
    
    dW = (np.dot(np.transpose(X), (sig_y_pred- y_new)))/y_new.shape[0]
    dW = dW + reg*W
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW
