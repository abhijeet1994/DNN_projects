from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    ############################################################################
    # TODO: Implement the affine forward pass. Store the result in 'out'. You  #
    # will need to reshape the input into rows.                      #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                   #
    ############################################################################
    out = np.dot(x,w) + np.transpose(b)
    
    ############################################################################
    #                    END OF YOUR CODE                    #
    ############################################################################
    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - x: input data, of shape (N, d_1, ... d_k)
    - w: weights, of shape (D, M)
    - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """
    ############################################################################
    # TODO: Implement the affine backward pass.                      #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                   #
    ############################################################################
    
    num = x.shape[0]
    x_re = x.reshape(num, -1)
    dx = np.dot(dout, np.transpose(w))
    dx = np.reshape(dx,x.shape)
    dw = np.dot(np.transpose(x_re), dout)
    db = np.sum(dout,axis=0)
    #db = np.dot(np.transpose(dout), np.ones(x.shape[0])) 
    ############################################################################
    #                    END OF YOUR CODE                    #
    ############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs).

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """
    ############################################################################
    # TODO: Implement the ReLU forward pass.                        #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                   #
    ############################################################################
    out = np.maximum(0,x)
    
    
    ############################################################################
    #                    END OF YOUR CODE                    #
    ############################################################################
    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs).

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """
    ############################################################################
    # TODO: Implement the ReLU backward pass.                       #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                   #
    ############################################################################
    
    dx2 = (x>0).astype(int)
    dx = dx2*dout
    
    ############################################################################
    #                    END OF YOUR CODE                    #
    ############################################################################
    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))
    
    Inputs:
    - x: (float) a tensor of shape (N, #classes)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradients wrt input x
    """
    ############################################################################
    # TODO: You can use the previous softmax loss function here.           # 
    # Hint: Be careful on overflow problem                         #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                   #
    ############################################################################
    
    N = y.shape[0]
    max_val = np.amax(x,axis = 1).reshape((N,1))
    x_new = x-max_val
    x_new = np.exp(x_new)
    x_sum = np.sum(x_new,axis = 1, keepdims =1)
    #x = np.exp(x - np.amax(x,axis = 1).reshape((N,1)))
    x_out = x_new/x_sum
    loss = np.log(x_out)
    loss = -1*np.sum(loss[range(N),y])/N
    dx = (x_out - np.eye(x.shape[1])[y])/N
    
   
    ############################################################################
    #                    END OF YOUR CODE                    #
    ############################################################################
    return loss, dx


    