# _*_ coding:utf-8 _*_
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import sys 
import numpy as np
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde
sys.setdefaultencoding('utf-8')   # 解决汉字编码问题

# import os
# sys.path.append(os.path.dirname(os.getcwd()))  # add the upper level directory into the pwd
# import tf_fun
import tensorflow as tf
import tensorflow.contrib.slim as slim
import Config as cfg0

import tf_fun

class SpliceNet:
    """
    the SpliceNet
    """
    def __init__(self,class_num, dropout_rate=0.2):
        """
        parameters:
           inputs:the input of the network
           is_training:trainable or not
           class_num:the final class number
           dropout_rate:dropout rate
        """ 
        self.class_num = class_num
        self.dropout_rate = dropout_rate 
        self.input_size = cfg0.image_column*cfg0.image_size
        self.output_size = cfg0.image_row*cfg0.image_column*self.class_num
        
        # construct placeholder
        self.inputs = tf.placeholder(tf.float32,[None, self.input_size, self.input_size, 3])  # the input of the network
        self.labels = tf.placeholder(tf.float32, [None, self.output_size])  # the labels of train sampels
        self.is_training = tf.placeholder(tf.bool)  # trainable or not
        
        self.tf_op = tf_fun.tf_fun(self.is_training)
        
        # build the network
        self.prob = self.build(self.inputs)
        
        # construct loss Function
        self.cost = -tf.reduce_mean(self.labels*tf.log(tf.clip_by_value(self.prob,1e-10,1.0)))  
        # self.cost = self.structCost()

#     def structCost(self):
#         """
#         function:
#             构造损失函数
#         paremeters：
#             None
#         """
#         cost = 0
#         image_row = cfg0.image_row
#         image_column = cfg0.image_column
#         assert list(self.prob.shape)[-1]==image_row*image_column*self.class_num, "网络输出节点的维度和配置文件不兼容！"
#         # prob_shape = self.prob.shape
#         # labels_shape = self.labels.shape
#         for i in range(0,image_row*image_column*self.class_num,self.class_num):
#             start = i
#             end = i+self.class_num 
#             prob_slice = tf.strided_slice(self.prob, [0, start], [cfg0.batch_size, end], [1, 1])  # prob切片  
#             prob_slice = tf.nn.softmax(prob_slice)
#             labels_slice = tf.strided_slice(self.labels,[0, start], [cfg0.batch_size, end], [1, 1])  # labels切片
#             cost_slice = -tf.reduce_mean(labels_slice*tf.log(tf.clip_by_value(prob_slice,1e-10,1.0)))  # 构造切片损失函数
#             cost = cost + cost_slice
#         return cost
    
    def bottleneck(self,bottom,conv33_channel1,conv33_channel2,block_name,stride=1):
        """
       function:
           the bottleneck of ResNet34
       parameters:
           bottom:the input tensor of the denseblock
           conv33_channel1:the number of the channel of the first conv(3×3) 
           conv33_channel2:the number of the channel of the second conv(3×3) 
           block_name:the name of the bottleneck
           strides：the stride of conv(1×1)
        """
        net1 = self.tf_op.conv_layer(bottom, conv33_channel1, kernel_size=3, stride=stride, layer_name=block_name+"/conv1",padding='SAME')
        net1 = self.tf_op.batch_normalization(net1, scope_name=block_name+"/bn1")
        net1 = tf.nn.relu(net1)
        
        net2 = self.tf_op.conv_layer(net1, conv33_channel2, kernel_size=3, stride=1, layer_name=block_name+"/conv2",padding='SAME')
        net2 = self.tf_op.batch_normalization(net2, scope_name=block_name+"/bn2")
        
        # Ensure that the shape of bottom and net2 are equal 
        net2_channel = net2.get_shape()[-1]
        if bottom.shape[1:4]!=net2.shape[1:4]:
            tmp = self.tf_op.conv_layer(bottom, net2_channel, kernel_size=1, stride=stride, layer_name=block_name+"/conv3",padding='SAME')
        else:
            tmp = bottom
        assert net2.shape[1:4]==tmp.shape[1:4], "net2 and tmp have different shapes！"
        
        # identity
        net3 = tf.add(net2,tmp)
        """
        # 注意：add操作之后不能使用BN，这里BN改变了“identity”分支的分布，影响了信息的传递，在训练的时候会阻碍loss的下降
        参考网址：https://blog.csdn.net/chenyuping333/article/details/82344334
        """
        # net3 = self.tf_op.batch_normalization(net3, scope_name=block_name+"/bn4") 
        net3 = tf.nn.relu(net3)
        return net3
    
    def build(self,inputs,scope="MobileNetV1"): 
        """
        function:
            build the ResNet34 network
        """
        assert inputs.get_shape().as_list()[1:]==[self.input_size,self.input_size,3], 'the size of inputs is incorrect!'
        
        # start to build the model
        net = inputs 
        print ("start-shape:"+str(net.shape))
        
        net = self.tf_op.conv_layer(net, kernel_num=64, kernel_size=7, stride=2, layer_name="conv1",padding='SAME')
        print ("the 0th stage-shape："+str(net.shape))
        
        net = self.tf_op.max_pool(net, layer_name="pool1", kernel_size=3, stride=2, padding='SAME')
        print ("the 1th stage-shape："+str(net.shape))
        
        # the first ResNet block
        for i in range(2):
            # s = 2 if i==0 else 1
            s = 1  # all the layers in the block with the stride 1
            net = self.bottleneck(net,64,64,block_name="bottleneck2_"+str(i),stride=s)
        print ("the 2th stage-shape："+str(net.shape))
        
        # the second ResNet block
        for i in range(2):
            s = 2 if i==0 else 1
            net = self.bottleneck(net,128,128,block_name="bottleneck3_"+str(i),stride=s)
        print ("the 3th stage-shape："+str(net.shape))
        
        # the third ResNet block
        for i in range(2):
            s = 2 if i==0 else 1
            net = self.bottleneck(net,256,256,block_name="bottleneck4_"+str(i),stride=s)
        print ("the 4th stage-shape："+str(net.shape))
        
        # the fourth ResNet block
        for i in range(2):
            s = 2 if i==0 else 1
            net = self.bottleneck(net,512,512,block_name="bottleneck5_"+str(i),stride=s)
        print ("the 5th stage-shape："+str(net.shape))
        
        # the fifth ResNet block
        for i in range(2):
            s = 2 if i==0 else 1
            net = self.bottleneck(net,256,256,block_name="bottleneck6_"+str(i),stride=s)
        print ("the 6th stage-shape："+str(net.shape))
        
        """
        # the sixth ResNet block
        for i in range(2):
            s = 2 if i==0 else 1
            net = self.bottleneck(net,512,512,block_name="bottleneck7_"+str(i),stride=s)
        print ("the 7th stage-shape："+str(net.shape))
        """
        
        net = self.tf_op.avg_pool(net, "avg_pool8", kernel_size=2, stride=2, padding='VALID')
        net_shape = net.get_shape() 
        net = self.tf_op.fc_layer(net, int(net_shape[1])*int(net_shape[2])*int(net_shape[3]), self.output_size, "fc9")
        print ("the 8th stage-shape："+str(net.shape))
        # result = tf.nn.softmax(net,name="prob")
        # 将总的softmax激活函数替换为每个细胞图片分组激活
        result = []
        for i in range(0,cfg0.image_row*cfg0.image_column*self.class_num,self.class_num):
            start = i
            end = i+self.class_num 
            prob_slice = tf.strided_slice(net, [0, start], [cfg0.batch_size, end], [1, 1])  # prob切片  
            prob_slice = tf.nn.softmax(prob_slice)
            result.append(prob_slice)
        result = tf.concat(result,axis=1)
        print ("the 10th stage-shape："+str(result.shape))
        return result
    
   