#!/usr/bin/env python
# ECBM E4040 Fall 2018 Assignment 2
# This script is intended for task 5 Kaggle competition. Use it however you want.

#!/usr/bin/env/ python
# ECBM E4040 Fall 2018 Assignment 2
# TensorFlow CNN

import tensorflow as tf
import numpy as np
import time
import csv
# from ecbm4040.image_generator import ImageGenerator

from utils.neuralnets.cnn.layers import conv_layer,max_pooling_layer, norm_layer, fc_layer

def my_LeNet(input_x, input_y = None, is_training= False,
          img_len=128, channel_num=3, output_size=10,
          conv_featmap=[6, 16], fc_units=[84],
          conv_kernel_size=[5, 5], pooling_size=[2, 2],
          l2_norm=0.01, seed=235,testvariable = False):
    """
        LeNet is an early and famous CNN architecture for image classfication task.
        It is proposed by Yann LeCun. Here we use its architecture as the startpoint
        for your CNN practice. Its architecture is as follow.

        input >> Conv2DLayer >> Conv2DLayer >> flatten >>
        DenseLayer >> AffineLayer >> softmax loss >> output

        Or

        input >> [conv2d-maxpooling] >> [conv2d-maxpooling] >> flatten >>
        DenseLayer >> AffineLayer >> softmax loss >> output

        http://deeplearning.net/tutorial/lenet.html

    """

    assert len(conv_featmap) == len(conv_kernel_size) and len(conv_featmap) == len(pooling_size)

    # conv layer
    conv_layer_0 = conv_layer(input_x=input_x,
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              rand_seed=seed)
    
    bn_0 = norm_layer(conv_layer_0.output(), is_training)
    
    pooling_layer_0 = max_pooling_layer(input_x=bn_0.output(),
                                        k_size=pooling_size[0],
                                        padding="VALID")
    
    # conv layer
    conv_layer_1 = conv_layer(input_x=pooling_layer_0.output(),
                              in_channel=conv_featmap[0],
                              out_channel=conv_featmap[1],
                              kernel_shape=conv_kernel_size[1],
                              rand_seed=seed,index=1)
    bn_1 = norm_layer(conv_layer_1.output(), is_training)
    pooling_layer_1 = max_pooling_layer(input_x=bn_1.output(),
                                        k_size=pooling_size[1],
                                        padding="VALID")
    
    
    
    
    conv_layer_2 = conv_layer(input_x=pooling_layer_1.output(),
                              in_channel=conv_featmap[1],
                              out_channel=conv_featmap[2],
                              kernel_shape=conv_kernel_size[2],
                              rand_seed=seed,index=2)
    bn_2 = norm_layer(conv_layer_2.output(), is_training)
    pooling_layer_2 = max_pooling_layer(input_x=bn_2.output(),
                                        k_size=pooling_size[2],
                                        padding="VALID")
    
    
    
    conv_layer_3 = conv_layer(input_x=pooling_layer_2.output(),
                              in_channel=conv_featmap[2],
                              out_channel=conv_featmap[3],
                              kernel_shape=conv_kernel_size[3],
                              rand_seed=seed,index=3)
    bn_3 = norm_layer(conv_layer_3.output(), is_training)
    pooling_layer_3 = max_pooling_layer(input_x=bn_3.output(),
                                        k_size=pooling_size[3],
                                        padding="VALID")
    
    
    # flatten
    pool_shape = pooling_layer_3.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(pooling_layer_3.output(), shape=[-1, img_vector_length])

    # fc layer
    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=0)
    
    bn_4 = norm_layer(fc_layer_0.output(), is_training)
       
    fc_layer_1 = fc_layer(input_x=bn_4.output(),
                          in_size=fc_units[0],
                          out_size=fc_units[1],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=1)
    
    bn_5 = norm_layer(fc_layer_1.output(), is_training)

    fc_layer_2 = fc_layer(input_x=bn_5.output(),
                          in_size=fc_units[1],
                          out_size=output_size,
                          rand_seed=seed,
                          activation_function=None,
                          index=2)

    conv_w = [conv_layer_0.weight,conv_layer_1.weight,conv_layer_2.weight,conv_layer_3.weight]
    fc_w = [fc_layer_0.weight, fc_layer_1.weight,fc_layer_2.weight]

    # loss
    if testvariable:
        return fc_layer_2.output() 
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        l2_loss += tf.reduce_sum([tf.reduce_sum(tf.norm(w, axis=[-2, -1])) for w in conv_w])

        label = tf.one_hot(input_y, 5)
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fc_layer_2.output()),
            name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('my_LeNet_loss', loss)

    return fc_layer_2.output(), loss



def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 5)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

    return ce


def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return step


def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('LeNet_error_num', error_num)
    return error_num

def evaluate2(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('LeNet_error_num', error_num)
    return pred



def predict(output):
    with tf.name_scope('output'):
        pred = tf.argmax(output, axis=1)
        tf.summary.scalar('pred_num', pred)
    return pred

####################################
#        End of your code          #
####################################

##########################################
# TODO: Build your own training function #
##########################################


# training function for the LeNet model
def my_training(X_train, y_train, X_val, y_val,x_test,y_test,
             conv_featmap=[6],
             fc_units=[84],
             conv_kernel_size=[5],
             pooling_size=[2],
             l2_norm=0.01,
             seed=235,
             learning_rate=1e-2,
             epoch=20,
             batch_size=245,
             verbose=False,
             pre_trained_model=None):
    print("Building my LeNet. Parameters: ")
    print("conv_featmap={}".format(conv_featmap))
    print("fc_units={}".format(fc_units))
    print("conv_kernel_size={}".format(conv_kernel_size))
    print("pooling_size={}".format(pooling_size))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, X_train.shape[1], X_train.shape[1], X_train.shape[-1]], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        is_training = tf.placeholder(tf.bool, name='is_training')

    output, loss = my_LeNet(xs, ys, is_training,
                         img_len=X_train.shape[1],
                         channel_num=X_train.shape[-1],
                         output_size=5,
                         conv_featmap=conv_featmap,
                         fc_units=fc_units,
                         conv_kernel_size=conv_kernel_size,
                         pooling_size=pooling_size,
                         l2_norm=l2_norm,
                         seed=seed)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss)
    eve = evaluate(output, ys)
    eve2 = evaluate2(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'lenet_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                raise ValueError("Load model Failed!")
                
        

        for epc in range(epoch):
            print("epoch {}".format(epc + 1))
            

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]
#                 print(X_train)
#                 print(training_batch_x)
                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x,
                                                                ys: training_batch_y,
                                                                is_training: True})

                
                
                if iter_total % 100 == 0:
                    # do validation
                    valid_eve,valid_eve2, merge_result = sess.run([eve,eve2, merge], feed_dict={xs: X_val,
                                                                                ys: y_val,
                                                                                is_training: False})
                    
                    valid_evetest,valid_eve2test, merge_resulttest = sess.run([eve,eve2, merge], feed_dict={xs: x_test,
                                                                                ys: y_test,
                                                                                is_training: False})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
#                     if verbose:
#                         print("train accuracy is {}".format(eve))
#                         print('{}/{} loss: {} validation accuracy : {}%'.format(
#                             batch_size * (itr + 1),
#                             X_train.shape[0],
#                             cur_loss,
#                             valid_acc))

                    # save the merge result summary
#                     writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        with open('predicted.csv','w') as csvfile:
                            fieldnames = ['Id','label']
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writeheader()    
                            for index,l2 in enumerate(valid_eve2test):
                                filename = str(index)+'.png'
                                label = str(l2)
                                writer.writerow({'Id': filename, 'label': label})
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))
    output = my_LeNet(input_x = x_test, testvariable = True)
    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))
##########################################
#            End of your code            #
##########################################

def my_testing(x_test, pre_trained_model=None):
    # define the variables and parameter needed during training
    with tf.Session() as sess:
        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                raise ValueError("Load model Failed!")
        output = my_LeNet(input_x = x_test, testvariable = True)
        pred = predict(output)
        
        predictedvalues = sess.run([pred], feed_dict={xs: x_test, is_training: False})    
        return predictedvalues
##########################################
#            End of your code            #
##########################################