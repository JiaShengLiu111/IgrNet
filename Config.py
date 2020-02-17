# _*_ coding:utf-8 _*_
# coding=utf-8
import sys 
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')   # 解决汉字编码问题


# the parameters of regular
weight_decay_rate = 1e-5  # L2正则化权重衰减系数
dropout_rate = 0.2  

# the parameters of train process
batch_size = 32
epochs = 160  # the total epochs of the total train_set
firstPred = True  # 表示在首次训练之前是否计算模型初始的损失函数
saveModelPath = '/notebooks/17_LJS/model2019/20190525_TBS三分类加入淡染样本/20190611_TBS三分类模型对比/SpliceNet新探索2_20190726/SpliceNet_增大图片拼接数量_单个图片SoftMax_按照不同样本拼接图片/model_1'  # 模型的保存路径
train_reDir = "info/info.txt"  # 训练过程输出信息重定向文件
test_redir = "info_test/info.txt"  # 预测过程输出信息重定向文件

# !!！注意!!！：一旦loadmodel_path则会加载预训练模型
loadmodel_path = '/notebooks/17_LJS/model2019/20190525_TBS三分类加入淡染样本/20190611_TBS三分类模型对比/SpliceNet新探索2_20190726/SpliceNet_增大图片拼接数量_单个图片SoftMax_按照不同样本拼接图片/'  # 预训练模型路径
loadmodel_name = 'model_1.meta'  # 预训练模型名称
saveStepEpochRate = 1  # 表示每训练saveStepEpochRate*epoch时，判断并保存一次最优模型

# the parameters of ptimizer
lr = 1e-3  # 初始学习率
lr_decay_times = 8  # （训练过程中）学习率衰减次数
lr_decay_rate = 0.94  # 学习率指数衰减率

# 拼接图像每行细胞个数、每列细胞个数
image_column=5
image_row=5
image_size = 112  # 单个宫颈细胞图像进行数据增强后的大小

junk_index = 0  # 表示在进行拼接时，当某一个样本的细胞数量不足以拼接时，背景区域当做成哪一个类别。

class_num = 3



# 修改备注： 
# 注意：getSingleTrainPic函数中取消了对列表混合打乱操作。！！！！！！！！！！！！！！！

  

