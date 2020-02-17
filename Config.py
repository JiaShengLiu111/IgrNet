# _*_ coding:utf-8 _*_
# coding=utf-8
import sys 
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')


# The parameters of L2 regular and dropout
weight_decay_rate = 1e-5  
dropout_rate = 0.2  

# The parameters of train process
batch_size = 32
epochs = 160  # the total epochs of the total train_set
firstPred = True  # evaluate the model at initial state or not
saveModelPath = 'xxx/model_1'  # save model path
train_reDir = "log/log.txt"  # the log file of training process

# The pretrained model
loadmodel_path = 'xxx'  # path of pretrained model
loadmodel_name = 'model_1.meta'  # name of pretrained model
saveStepEpochRate = 1  # evaluate the model every saveStepEpochRate EPOCH

# The parameters of optimizer
lr = 1e-3  # init learning rate
lr_decay_times = 8  # the times of lr exponential decay
lr_decay_rate = 0.94  # the rate of lr exponential decay

# The parameters of IgrNet
image_column=5  # the column of spliced image
image_row=5  # the row of spliced image
image_size = 112  # the size of cell patches
class_num = 3  # the number of category
junk_index = 0  # the category of padding images


