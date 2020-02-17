# _*_ coding:utf-8 _*_
# coding=utf-8
import sys 
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')   # 解决汉字编码问题

import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm
import tensorflow.contrib.slim as slim

"""
funtion list: 
    conv、
    full-connection、
    pool(max_pool,avg_pool,global_avg_pool,global_avg_pool)、
    activation-function、
    dropout、
    bn、
    concat、
    depthwise_separable_conv
    
    etc.
"""

class tf_fun:
    """
    common tensorflow funtion used to build model.
    """
    def __init__(self,is_training):
        """
        funtion:
            init funtion
        parameters:
            is_training:trainable or not
        """
        self.is_training = is_training
        pass
    
    def conv_layer(self,bottom, kernel_num, kernel_size, stride=1, layer_name="conv",padding='SAME'):
        """
        funtion:
            convolution funtion 
        parameters：
            bottom:the input feature map
            kernel_num:the number of convolution kernel
            kernel_size:the size of convolution kernel
            stride:the stride of convolution
            layer_name:the name of the layer
            padding:the style of the convolution
        return:the result of convoluetion
        """
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=bottom, filters=kernel_num, kernel_size=kernel_size, strides=stride, padding=padding)
            return network
        
    def fc_layer(self, bottom, in_size, out_size, layer_name):
        """
        funtion:
            full-connection funtion 
        parameters：
            bottom:the input feature map
            in_size:the size of input
            out_size:the size of output
            layer_name:the name of the layer 
        return:the result of full-connection
        """
        with tf.variable_scope(layer_name):
            weights, biases = self.get_fc_var(in_size, out_size, layer_name) 
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases) 
            return fc
        
    def get_fc_var(self, in_size, out_size, name):
        """
        funtion:
            get the weights and biases with the designated(指定的) size 
        """
        # weights
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        var_name = name + "_weights"
        weights = tf.Variable(initial_value, name=var_name)
        # biases
        initial_value = tf.truncated_normal([out_size], .0, .001)
        var_name = name + "_biases"
        biases = tf.Variable(initial_value, name=var_name)
        return weights, biases
    
    def avg_pool(self, bottom, layer_name, kernel_size=2, stride=2, padding='SAME'):
        """
        function:
            avg_pool
        parameters:
            bottom:the input feature map
            kernel_size:the number of pool convolution kernel
            stride:the stride of the pool
            name:the name of pool layer
            padding:the type of the pool
        """
        return tf.nn.avg_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=layer_name)

    def max_pool(self, bottom, layer_name, kernel_size=2, stride=2, padding='SAME'):
        """
        function:
            max_pool
        parameters:
            bottom:the input feature map
            kernel_size:the number of pool convolution kernel
            stride:the stride of the pool
            name:the name of pool layer
            padding:the type of the pool
        """
        return tf.nn.max_pool(bottom, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=layer_name) 
    
    def global_avg_pool(self, bottom, layer_name, stride=1):
        """
       function:
           golbal average pooling
       parameters:
           same as function avg_pool.
       attention: padding is 'VALID' for global_pool
        """
        shape=bottom.get_shape()  # get the shape of bottom:(None,Row,Col,channel)
        result = tf.nn.avg_pool(bottom, ksize=[1, shape[1], shape[2], 1], strides=[1, stride, stride, 1], padding='VALID', name=layer_name)
        return result
    
    def global_max_pool(self, bottom, layer_name, stride=1): 
        """ 
       function:
           golbal max pooling
       parameters:
           same as function max_pool. 
       attention: padding is 'VALID' for global_pool
        """
        shape=bottom.get_shape()  # get the shape of bottom:(None,Row,Col,channel)
        return tf.nn.max_pool(bottom, ksize=[1, shape[1], shape[2], 1], strides=[1, stride, stride, 1], padding='VALID', name=layer_name)
    
    def drop_out(self, x, dropout_rate=0.2) :
        """
        function:
            dropout
        parameters:
            rate:the probability of doupout(keep_prob=1-rate) 
        """
        return tf.layers.dropout(inputs=x, rate=dropout_rate, training=self.is_training)
    
    def batch_normalization(self, bottom, scope_name):
        """
       function:
           batch normalization 
        """
        with arg_scope([batch_norm],
                       scope=scope_name,
                       updates_collections=None,
                       decay=0.9,
                       center=True,
                       scale=True,
                       zero_debias_moving_mean=True,
                       reuse=tf.AUTO_REUSE) :
            return batch_norm(inputs=bottom, is_training=self.is_training)
        
    def concat(self,bottom,axis=3):
        """
        function:
            Concatenation
        parameters:
            bottom:the list of tensor waitting to concat
            axis:the concat channel 
        remarks:
            the bottom often is feature map which shaped like (None,Row,Col,Channel)，so the default value of axis is 3
            represent that concat bottom from the 'channel'.
        """
        return tf.concat(bottom,axis=axis)

    def depthwise_separable_conv(self,bottom,num_pwc_filters
                                  ,width_multiplier,strides,scopename):
        """
        function:
            Function to build the depth-wise separable convolution layer.
        parameters:
            bottom:the list of tensor waitting to concat
            num_pwc_filters：the base channel of pointwise_convolution
            width_multiplier：the scale factor of the base channel of pointwise_convolution,\
                        namely, the channel of output is num_pwc_filters×width_multiplier：the
        strides：the stride of depthwise_convolution
        scopename：the name of depthwise_separable_conv layer
        """

        # the really channel of the final output
        num_pwc_filters = round(num_pwc_filters * width_multiplier)  
        """
        depthwise convolution
        Note1：skip pointwise by setting num_outputs=None
        Note2：the separable_convolution2d don't need input the number of convolution kernels,\
        because the number of convolution kernels equals to the channel of bottom  
        """
        depthwise_conv = slim.separable_convolution2d(bottom\
                                         ,num_outputs=None,stride=strides,depth_multiplier=1\
                                         ,activation_fn=None,padding='SAME'
                                         ,kernel_size=[3, 3],scope=scopename+'/depthwise_conv')
        bn = self.batch_normalization(depthwise_conv,scopename+"/dw_batch_norm")
        bn = tf.nn.relu(bn)

        # pointwise convolution
        pointwise_conv = self.conv_layer(bn, kernel_num=num_pwc_filters, kernel_size=1, stride=1\
                               , layer_name=scopename+'/pointwise_conv',padding='SAME')
        bn = self.batch_normalization(pointwise_conv,scopename+"/pw_batch_norm")
        bn = tf.nn.relu(bn)
        return bn

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    