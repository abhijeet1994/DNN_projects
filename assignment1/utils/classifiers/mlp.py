from builtins import range
from builtins import object
import numpy as np

from utils.layer_funcs import *
from utils.layer_utils import *

class MLP(object):
    """
    MLP with an arbitrary number of dense hidden layers,
    and a softmax loss function. For a network with L layers,
    the architecture will be

    input >> DenseLayer x (L - 1) >> AffineLayer >> softmax_loss >> output

    Here "x (L - 1)" indicate to repeat L - 1 times. 
    """
    def __init__(self, input_dim=3072, hidden_dims=[200,200], num_classes=10, reg=0.0, weight_scale=1e-3):
        """
        Inputs:
        - reg: (float) L2 regularization
        - weight_scale: (float) for layer weight initialization
        """
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        
        dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(dims)-1):
            layers.append(DenseLayer(input_dim=dims[i], output_dim=dims[i+1], weight_scale=weight_scale))
        layers.append(AffineLayer(input_dim=dims[-1], output_dim=num_classes, weight_scale=weight_scale))
        
        self.layers = layers

    def loss(self, X, y):
        """
        Calculate the cross-entropy loss and then use backpropogation
        to get gradients wst W,b in each layer.
        
        Inputs:
        - X: input data
        - y: ground truth
        
        Return loss value(float)
        """
        loss = 0.0
        reg = self.reg
        num_layers = self.num_layers
        layers = self.layers
        ####################################################
        # TODO: Feedforward                      #
        ####################################################
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################
        input_next = X
        #for layer in self.layers:
        #    input_next = layer.feedforward(input_next)
        for i in range(len(self.layers)):
            input_next = self.layers[i].feedforward(input_next)
        
        loss, dout = softmax_loss(input_next, y)
        
        
        ####################################################
        # TODO: Backpropogation                   #
        ####################################################
        dX = dout
        for x in self.layers[::-1]:
            dX = x.backward(dX)
        ####################################################
        # TODO: Add L2 regularization               #
        ####################################################
        square_weights = np.sum([np.sum(x.params[0]**2) for x in self.layers])
        loss += 0.5*self.reg*square_weights
        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
        
        return loss

    def step(self, learning_rate=1e-5):
        """
        Use SGD to implement a single-step update to each weight and bias.
        """
        params = []
        grads = []
        layers = self.layers
        num_layers = self.num_layers
        for  layer in layers:
            params.extend(layer.params)
            grads.extend(layer.gradients)
        # Add L2 regularization
        reg = self.reg
        grads = [grad + reg*params[i] for i, grad in enumerate(grads)]
        ####################################################
        # TODO: Use SGD to update variables in layers.    #
        ####################################################
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################
        #for i in range(len(params)): 
        #    params[i] -=learning_rate*grads[i]
        params = [params[i] - learning_rate*grads[i] for i in range(len(grads))]
        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
   
        # update parameters in layers
        for i in range(num_layers):
            self.layers[i].update_layer(params[2*i:2*(i+1)])
        

    def predict(self, X):
        """
        Return the label prediction of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        
        Returns: 
        - predictions: (int) an array of length N
        """
        predictions = None
        num_layers = self.num_layers
        layers = self.layers
        #####################################################
        # TODO: Remember to use functions in class       #
        # SoftmaxLayer                          #
        #####################################################
        ####################################################
        #           START OF YOUR CODE           #
        ####################################################
        output  = X
        for x in layers:    
            output = x.feedforward(output)
        out2 = np.exp(output)
        output_softmax = out2/np.sum(out2,axis = 1,keepdims =1)
        predictions = np.argmax(output_softmax,axis = 1)
        ####################################################
        #            END OF YOUR CODE            #
        ####################################################
        
        return predictions
    
    def check_accuracy(self, X, y):
        """
        Return the classification accuracy of input data
        
        Inputs:
        - X: (float) a tensor of shape (N, D)
        - y: (int) an array of length N. ground truth label 
        Returns: 
        - acc: (float) between 0 and 1
        """
        y_pred = self.predict(X)
        acc = np.mean(np.equal(y, y_pred))
        
        return acc