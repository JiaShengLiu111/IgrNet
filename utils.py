# _*_ coding:utf-8 _*_
# coding=utf-8
import sys 
stdi,stdo,stde=sys.stdin,sys.stdout,sys.stderr 
reload(sys) 
sys.stdin,sys.stdout,sys.stderr=stdi,stdo,stde 
sys.setdefaultencoding('utf-8')   # 解决汉字编码问题

import skimage
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from PIL import Image
import time
import random
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score   
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score  
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
import os

class DataEnhance():
    """
    python实现（图片）数据增强
    参考网址：
    1、https://www.jb51.net/article/45653.htm（产生随机数）
    2、https://blog.csdn.net/guduruyu/article/details/70842142
    3、https://blog.csdn.net/qq_23301703/article/details/79908988
    """
    def __init__(self,image_h = 112,image_w = 112,crop_rate=0.875):
        """
       image_h,image_w用于（通过resize函数）限定图片的长和宽 
        """
        self.image_h = image_h
        self.image_w = image_w
        self.crop_rate = crop_rate
        pass

    def randomCrop(self,img,rate):     
        """
       function:根据比率从图片中随机截取子图
       parameters:
           img:待截取图像（默认输入图像长宽相等）
           rate:截取的比率
       return:待返回图像crop_img
        """
        length = np.array(img).shape[0]
        resizeLength = int(length*rate) # crop_img的边长
        rangeLength = length - resizeLength  # 原图和crop_img边长差
        # 随机产生crop_img的左上角坐标
        cropX = random.randint(0, rangeLength)  
        cropY = cropX 
        # 根据crop_img左上角坐标和右下角坐标截取图像
        box = (cropX, cropY, cropX+resizeLength, cropY+resizeLength)  
        crop_img = img.crop(box)   
        return crop_img

    def randomFlipLeftRight(self,img):    
        """
       function:
           对img图片进行随机的水平翻转
       parameters：
           img:待处理图片
        """
        randNum = random.randint(0, 1)
        if randNum%2==0: 
            h_flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
        else:
            h_flip_img = img
        return h_flip_img 

    def randomFlipUpDown(self,img):  
        """
       function:
           对img图片进行随机的垂直翻转
       parameters：
           img:待处理图片
        """
        randNum = random.randint(0, 1)
        if randNum%2==0: 
            v_flip_img = img.transpose(Image.FLIP_TOP_BOTTOM)   # 进行垂直翻转
        else:
            v_flip_img = img
        return v_flip_img  

    def randomLightTrans(self,img):
        """
        function:对img进行随机光强变换
        parameters:待处理图片

        """
        light_factors = [0.75 + i * 0.05 for i in range(10)]
        light_factor = random.choice(light_factors)
        img = img.point(lambda i : i * light_factor)
        return img

    def randomRotate(self,img):
        """
       function:对img进行随机360度旋转
       parameters:
           img:待进行随机旋转图片
        """
        randNum = random.randint(0, 360)  # 生成随机数（0~360范围）  
        img = img.rotate(angle=randNum, resample=0, expand=0)
        return img
    
    def dataEnhance(self,img,rate):
        """
       function:
           对输入图片进行随机截取、随机水平翻转、随机竖直翻转、随机光强变换
       parameters:
           img:待进行数据增强图片
        """
        x = self.randomCrop(img,rate)  # 随机截取
        x = self.randomFlipLeftRight(x)  # 随机水平翻转
        x = self.randomFlipUpDown(x)  # 随机垂直翻转
        x = self.randomLightTrans(x)  # 随机光强变换
        x = self.randomRotate(x)  # 随机旋转角度
        return x
    
    def centerImageCrop(self,img,rate):
        """
       function: 
           截取一张图片最中间的ROI图，边长是原始边长的rate倍
       parameters:
           img:待截取图像（默认输入图像长宽相等）
           rate:截取的比率
       return:待返回图像crop_img
        """
        length = np.array(img).shape[0]
        resizeLength = int(length*rate)
        # 生成crop_img图片左上角坐标
        yy = int(length*(1-rate)/2)
        xx = yy 
        box = (yy,xx,yy+resizeLength,xx+resizeLength)  # 根据box截取图像
        crop_img = img.crop(box)
        return crop_img
    
    def loadAndEnhanceTrainData(self,path): 
        """
        function:
            加载训练图片，并进行数据增强 
        parameters:
            path:图片路径
        """
        img = Image.open(path)  # load image  
        img = self.dataEnhance(img,self.crop_rate) # 数据增强（随机截取，随机水平翻转，随机竖直翻转，随机光强变换）
        img = np.array(img)
        img = img / 255.0 # 最大值归一化
        assert (0 <= img).all() and (img <= 1.0).all()
        # 限定图片的大小
        resized_img = skimage.transform.resize(img, (self.image_h, self.image_w)) 
        return resized_img 

    def loadAndEnhanceTestData(self,path): 
        """
        function:
            加载测试图片
        parameters:
            path:图片路径
        """
        img = Image.open(path)  # load image 
        img = self.centerImageCrop(img,self.crop_rate) # 根据rate从图片的中心截取一个ROI图像
        img = np.array(img)
        img = img / 255.0 # 对图片进行最大值归一化
        assert (0 <= img).all() and (img <= 1.0).all()
        # 限定图片的大小
        resized_img = skimage.transform.resize(img, (self.image_h, self.image_w)) 
        return resized_img 
    
    def getMiniBatch4TestAndVal(self,batchDir): 
        """
        function:
            读取一个batch的测试图片
        parameters:
            batchDir:batch图片路径
        """
        allSamples = []
        allSamplesDir = batchDir
        for i in range(len(allSamplesDir)):
            img_tmp = self.loadAndEnhanceTestData(allSamplesDir[i])
            allSamples.append(img_tmp) 
        allSamples = np.array(allSamples) 
        return allSamples

    def getMiniBatch4Train(self,batchDir):
        """
        function:
            读取一个batch的训练图片
        parameters:
            batchDir:batch图片路径
        """
        allSamples = []
        allSamplesDir = batchDir
        for i in range(len(allSamplesDir)):
            img_tmp = self.loadAndEnhanceTrainData(allSamplesDir[i])
            allSamples.append(img_tmp) 
        allSamples = np.array(allSamples)
        return allSamples    

    
class utils():
    """
    读取图片、调用模型进行预测、计算样本真实标签、绘制准确率/损失函数变化曲线等。
    """
    def __init__(self): 
        pass
    
    def printRd(self,value,filepath='info/info.txt',mode="a+"):  
        """
        function:
            print函数输出重定向到文件。
        parameters:
            value: str 要重定向输出的内容
            filepath: str 重定向文件路径
            model: str 重定向内容写入文件的模式。ref:https://www.runoob.com/python/python-func-open.html
        """
        f=open(filepath,mode)
        print >> f,value 
        f.close()

    def load_image(self,path):
        """
        function:
            读取图片，并以该图片短边为边长从该图片中截取子图，作为函数返回结果
        parameters:
            path:图片路径
        """
        img = skimage.io.imread(path)  # load image
        img = img / 255.0  # 最大值归一化
        assert (0 <= img).all() and (img <= 1.0).all() 
        short_edge = min(img.shape[:2])  # 计算图片短边
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]  # 截取图片
        resized_img = skimage.transform.resize(crop_img, (224, 224))  # resize to 224, 224
        return resized_img
    
    def onehot2realLabel(self,probs_,trainCountSample=None):
        """
       function:
           probs是一个列表，其包含的元素prob是形式上类似于onehot形式的模型预测结果，
           该方法依次对probs每一个元素进行如下处理：
               a.首先对prob进行训练集类别数量均衡处理
               b.然后求均衡结果中最大值所在的索引下标
       parameters:
           probs:形如onehot形式的模型预测结果所构成的列表
           trainCountSample:训练集中各类别样本的数量，形如[11,222,334]
        """
        probs = list(probs_)[:]  # 深拷贝
        # 首先进行训练集各类别数量均衡处理 
        if trainCountSample!=None:
            assert len(probs)>0, "onehot2realLabel输入空列表"
            assert len(probs[0])==len(trainCountSample), "onehot2realLabel中模型预测类别数和训练集样本真实类别数不相等"
            for i in range(len(probs)):
                prob = probs[i]  
                for j in range(len(prob)): 
                    prob[j] = prob[j]/trainCountSample[j]
        
        # 求probs中元素的最大值索引下标
        result = []
        for i in range(len(probs)):
            prob = list(probs[i])
            max_index = prob.index(max(prob))
            result.append(max_index)
        result = np.array(result)
        return result 

    # show performanceCurve
    def showPerformOrCostCurve(self,lists,names,picFullPath):
        """
       function:
           绘制模型训练曲线：准确率、F1分数、损失函数等 
           参考网址：
               https://blog.csdn.net/whjstudy1/article/details/80484613
       parameters:
           lists:二维列表，每一个维度表示某一个动态变化的数据，例如准确率或F1分数或损失函数，和names列表中元素一一对应
           names:一维列表，和lists中的元素一一对应，表示lists中某一维度数据所表示的含义。
           picFullPath:绘制的曲线的保存路径。
        """
        assert len(lists)==len(names),"showPerformOrCostCurve中数据的长度和标签的长度不相等"
        for i in range(len(lists)):  # 依次对lists中的每一维数据进行处理
            # 获取某一维数据的值和名称
            list_ = lists[i]
            name = names[i] 
            x_index = [i for i in range(len(list_))]  # 所绘制的二维曲线的x坐标
            plt.plot(x_index,list_,linewidth=2,label=name)  # 绘制曲线
        plt.legend(loc="best")  # 指定曲线标签的位置
        plt.title("performance curve",fontsize=24) 
        # plt.show()  # 显示图片 
        plt.savefig(picFullPath)  # 保存图片
        plt.clf()  
        plt.close() 

    def predicts(self,sess,sess_op,inputs_placeholder,is_training_placeholder,X_input,batch_size,dataenhance,reDir):
        """
       functions:
           调用tf模型对X_input进行预测，返回预测结果，并统计预测所消耗的时长
       patameters:
           sess:当前模型所在的Session
           sess_op:待计算的模型最后一层的预测结果
           inputs_placeholder:待预测样本的placeholder
           is_training_placeholder:表示训练状态的placeholder
           X_input:待预测样本的路径集合
           batch_size:对X_input进行逐batch预测所采用的batch_size大小
           dataenhance:DataEnhance实例，用于读取样本
           reDir:输出信息文件
       returns:
           X_input所对应的预测结果列表、预测所有样本所消耗的总时间
        """
        ReadDataTime = 0  # 用于记录访问磁盘读取数据的时间
        time_start = time.time()  # 起始时间
        result = []
        for i in range(0,len(X_input),batch_size):
            if i%100==0:
                self.printRd("myPredicts:"+str(i),reDir)
            start = i
            end = min(start+batch_size,len(X_input))  
            read_start = time.time()
            XX = dataenhance.getMiniBatch4TestAndVal(X_input[start:end])  # 访问磁盘读取文件耗时
            read_end = time.time()
            read_time = read_end - read_start
            ReadDataTime = ReadDataTime + read_time  # 累计读取磁盘数据的时间 
            result_tmp = sess.run(sess_op,feed_dict={inputs_placeholder:XX,is_training_placeholder:False}) 
            result_tmp = list(result_tmp)
            result = result+result_tmp 
        time_end = time.time()
        return np.array(result),time_end-time_start-ReadDataTime
   
    def analyResult(self,labels,predicts):
        """
       function:
           根据样本真实标签和预测结果计算：混淆矩阵、准确率、精确率、召回率、F1分数等。
       parameters:
           labels:样本的真实标签
           predicts:模型预测结果
        """
        resultsAnaly = confusion_matrix(labels,predicts)
        print("混淆矩阵：")
        print(resultsAnaly) 

        accuracy = accuracy_score(labels, predicts, normalize=True)
        precision_tmp = precision_score(labels, predicts,average="macro")
        f1_score_tmp = f1_score(labels, predicts,average="macro")
        recall_score_tmp = recall_score(labels,predicts,average='macro')

        print("准确率："+str(round(accuracy,5)))
        print("精确率："+str(round(precision_tmp,5)))
        print("召回率："+str(round(recall_score_tmp,5)))
        print("f1值： "+str(round(f1_score_tmp,5))) 
        return accuracy,precision_tmp,recall_score_tmp,f1_score_tmp
    
    def myShuffle(self,inputs,random_state=888):
        """
       function:
           对inputs进行乱序排列——使用train_test_split实现
       parameters:
           inputs:待乱序列表
           random_state:种子点
        """
        X_train,_,_,_ = train_test_split(inputs,[1]*len(inputs),test_size = 0,random_state =random_state)
        return X_train

    def countSample(self,inputs,labels):
        """
        function:
            统计数据集类别数量分布
        parameters:
            inputs:样本路径列表
            labels:类别标签列表，长度等于分类类别数
        """
        result = []
        for i in range(len(labels)):
            result.append(0)
        for i in range(len(inputs)):
            value = inputs[i]
            flag = 0  # 标识value是否能够在labels中找到相应的标签
            for j in range(len(labels)):
                if value.find(labels[j])>=0:
                    result[j]=result[j]+1
                    flag=1
                    break
            assert flag!=0, "countSample函数中出现了无法解决的bug！"  # 表示value无法找到相应的标签
        # 打印样本统计信息
        for i in range(len(labels)):
            print("类别"+labels[i]+"样本数为：\t"+str(result[i]))
        print("\n")
        return result
    
    def geneLabel(self,inputs,labels):
        """
        function:
            为样本生成one-hot标签
        parameters:
            inputs:样本路径列表
            labels:类别标签列表，长度等于分类类别数
        """
        y = []
        for i in range(len(inputs)):
            value = inputs[i]
            flag = 0  # 标识value是否能够在labels中找到相应的标签
            for j in range(len(labels)):
                if value.find(labels[j])>=0:
                    value_y = [0]*len(labels)  # 生成one-hot标签
                    value_y[j] = 1
                    y.append(value_y)
                    flag=1
                    break
            if flag==0:  # 表示value无法找到相应的标签
                print("geneLabel函数中出现了无法解决的bug！") 
        return y
    
    def random_without_same(self,mi, ma, num):
        """
       function:
           生成一定范围内指定数目的无重复数
       parameters:
           mi:区间下限
           ma:区间上限
           num:生成数目的数量
        """
        temp = range(mi, ma)  # 生成[mi,ma]之间的所有数
        # random.shuffle(temp)  # 对数据乱序排列（无法指定种子点）
        temp = self.myShuffle(temp) 
        return temp[0:num]  # 返回数组前num个数字

    def randomCropSample(self,X,y,num):
        """
       function:
           随机从某样本列表中截取一定数量无重复的样本
       parameters:
          X:待截取的列表（样本）
          y:待截取的类表（标签）
          num:截取的样本数量
        """
        randIndex_inTrain = self.random_without_same(0,len(X),num)
        X_crop = []
        y_crop = []
        for i in range(len(randIndex_inTrain)):
            index = randIndex_inTrain[i]
            X_crop.append(X[index])
            y_crop.append(y[index])
        return X_crop,y_crop
    
    def getSamplesDir(self,filepath,labels):
        """
        function:
            返回所有子目录labels[x]下所有文件的路径
        parameter:
            filepath:表示父目录
            labels:表示子目录列表
        """
        result = []
        for i in range(len(labels)):  # 依次读取每一个类别的样本路径
            label_name = labels[i]
            label_path = filepath+label_name
            allfile_name = os.listdir(label_path)  # 获取该类别所有样本名称
            allfile_fullpath = [os.path.join(label_path,allfile_name[i]) for i in range(len(allfile_name))]  # 该类别所有样本全路径
            # result.append(allfile_fullpath)
            result = result + list(allfile_fullpath)
        return result
    
    def writeCurveValue(self,lists,names,txtFullPath):
        """
       function:
           将模型训练曲线值写入文件：准确率、F1分数、损失函数等的列表总长度、最大值、最小值等。
       parameters:
           lists:二维列表，每一个维度表示某一个动态变化的数据，例如准确率或F1分数或损失函数，和names列表中元素一一对应
           names:一维列表，和lists中的元素一一对应，表示lists中某一维度数据所表示的含义。
           txtFullPath:保存文件的全路径。
        """
        assert len(lists)==len(names),"writeCurveValue中数据的长度和标签的长度不相等"
        self.printRd("",txtFullPath,"w+")  # 清空txtFullPath文件
        
        for i in range(len(lists)):  # 依次对lists中的每一维数据进行处理
            # 获取某一维数据的值和名称
            list_ = lists[i]
            name = names[i] 
            max_ = max(list_)  # 获取该维度的最大值
            max_index = list(list_).index(max_)  # 获取最大值所在的索引
            min_ = min(list_)  # 获取该维度的最小值
            min_index = list(list_).index(min_)  # 获取最小值所在的索引
            
            # 写入文件（追加模式）
            self.printRd(name+":",txtFullPath,"a+")  # 写入列表名
            self.printRd(str(list_),txtFullPath,"a+")  # 写入列表值
            self.printRd("length="+str(len(list(list_))),txtFullPath,"a+")  # 写入列表总长度
            self.printRd("max_index="+str(max_index)+"   max="+str(max_),txtFullPath,"a+")  # 写入列表最大值及所在的索引 
            self.printRd("min_index="+str(min_index)+"   min="+str(min_),txtFullPath,"a+")  # 写入列表最小值及所在的索引 
            self.printRd("\n",txtFullPath,"a+")
        return 0

if __name__ == "__main__":
    test()
