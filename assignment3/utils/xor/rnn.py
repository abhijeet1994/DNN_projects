#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
class MyLSTMCell(RNNCell):
    """
    Your own basic LSTMCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow LSTMCell source code. It's located at tensorflow/tensorflow/python/ops/rnn_cell_impl.py.
    If you're using Anaconda, it's located at
    anaconda_install_path/envs/your_virtual_environment_name/site-packages/tensorflow/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow LSTMCell, but with your own language.
    Also, you will find Colah's blog about LSTM to be very useful:
    http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """

    def __init__(self, num_units, num_proj, forget_bias=1.0, activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the LSTM cell.
        :param num_proj: The output dimensionality. For example, if you expect your output of the cell at each time step to be a 10-element vector, then num_proj = 10.
        :param forget_bias: The bias term used in the forget gate. By default we set it to 1.0.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyLSTMCell, self).__init__(_reuse=True)
        #############################################
        #           TODO: YOUR CODE HERE            #
        self.num_units = num_units
        self.num_proj = num_proj
        self.forget_bias = forget_bias
        
        
        if activation:
          self.activation = activations.get(activation)
        else:
          self.activation = math_ops.tanh
        #tf.random.set_seed(1355)
        ##################SETTING UP VARIABLES
        ###FOR INPUT LAYER
        self.inp_weights = tf.get_variable("inp_weights", [1+self.num_proj, self.num_units] , initializer=tf.random_normal_initializer())
        self.inp_bias = tf.get_variable("inp_bias", [1, self.num_units] , initializer=tf.constant_initializer(0.0))
        ##For Update Layer
        self.up_weights_1 = tf.get_variable("up_weights_1", [1+self.num_proj, self.num_units] , initializer=tf.random_normal_initializer())
        self.up_weights_2 = tf.get_variable("up_weights_2", [1+self.num_proj, self.num_units] , initializer=tf.random_normal_initializer())
        self.up_bias_1 = tf.get_variable("up_bias_1", [1, self.num_units] , initializer=tf.constant_initializer(0.0))
        self.up_bias_2 = tf.get_variable("up_bias_2", [1, self.num_units] , initializer=tf.constant_initializer(0.0))
        ## for output layer
        self.out_weights_1 = tf.get_variable("out_weights_1", [1+self.num_proj, self.num_proj] , initializer=tf.random_normal_initializer())
        self.out_weights_2 = tf.get_variable("out_weights_2", [self.num_units, self.num_proj] , initializer=tf.random_normal_initializer())
        self.out_bias_1 = tf.get_variable("out_bias_1", [1, self.num_proj] , initializer=tf.constant_initializer(0.0))
        self.out_bias_2 = tf.get_variable("out_bias_2", [1, self.num_proj] , initializer=tf.constant_initializer(0.0))
        ##################VARIABLES SET UP
        #############################################
        #raise NotImplementedError('Please edit this function.')
            
    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units + output_size

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        self._state_size = self.num_units + self.num_proj
        return self._state_size
        #############################################
        #raise NotImplementedError('Please edit this function.')

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        return self.num_proj
        #############################################
        #raise NotImplementedError('Please edit this function.')


    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step, calculate the current state and cell output.

        You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the very basic LSTM functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function state_size(self).
        :return: A tuple containing (output, new_state). For details check TensorFlow LSTMCell class.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        sigmoid = math_ops.sigmoid
        c, h = array_ops.split(state, [64,2], 1) #use slice
        ###### Forget Layer
        new_input = array_ops.concat([inputs, h], 1)
        forget_layer = math_ops.add(math_ops.matmul(new_input, self.inp_weights) , self.inp_bias) 
        c = math_ops.multiply(c, sigmoid(forget_layer))
        
        ###### Update Layer
        up_layer_1 = sigmoid(math_ops.add(math_ops.matmul(new_input, self.up_weights_1) , self.up_bias_1))
        up_layer_2 = self.activation(math_ops.add(math_ops.matmul(new_input, self.up_weights_2) , self.up_bias_2))
        update = math_ops.multiply(up_layer_1,up_layer_2)
        c = math_ops.add(c , update)
        
        ####### Output Layer
        out_layer_1 = sigmoid(math_ops.add(math_ops.matmul(new_input, self.out_weights_1) , self.out_bias_1))
        out_layer_2 = self.activation(math_ops.add(math_ops.matmul(c, self.out_weights_2) , self.out_bias_2))
        h = math_ops.multiply(out_layer_1, out_layer_2)
        
        #comp_inputs = array_ops.concat([inputs, h], 1)
        state = array_ops.concat([c, h], 1)
        return (h, state)
        #############################################
        #raise NotImplementedError('Please edit this function.')