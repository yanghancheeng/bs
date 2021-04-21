# from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.utils import to_categorical
from define import comple_listdir
import itertools
import cv2
import os
import pandas as pd
import numpy as np
import random
from define import get_str

def getModel(sel,continue_train,input_shape,filepath):

    def initMode():
        if sel == 5:
            model = get_Xception(input_shape, out_node)  # Xecp
        elif sel == 4:
            model = get_VGG(1, input_shape, out_node)  # mask_aug
        elif sel == 3:
            model = get_VGG(1, input_shape, out_node)  # mask
        elif sel == 2:
            model = get_VGG(1, input_shape, out_node)  # aug
        elif sel == 1:
            model = get_VGG(1, input_shape, out_node)  # ImageNet
        else:
            model = get_VGG(0, input_shape, out_node)

    out_node=10

    if continue_train==1 and os.access(filepath, os.F_OK):
        mdf = get_str('读入权重文件（完整路径）',filepath)
        return load_model(mdf)
    else:
        if continue_train==1:
            print("接上次模型，但模型文件丢失，使用随机权重")
        return initMode()

# 貌似用不到getImageArr
def myAugDataGenerator(DataFrame,exptype,Aug,batch_size,input_height, input_width,incl_preimg=True):
    DataFrame = DataFrame.sample(frac=1)  #随机打乱顺序
    zipped = itertools.cycle(zip(DataFrame.pic_path))
    
    #无数据增强模式
    if exptype==1:
        while True:
            IMG = []
            LABEL = []
            for _ in range(batch_size):
                im = next(zipped)[0]
                index = DataFrame.index[(DataFrame.pic_path==im)].tolist()[0]  #索引值
                img = cv2.resize(cv2.imread(im), (input_width, input_height))
                IMG.append(img/ 127.5 - 1)  #默认float32格式
                LABEL.append(np.array(eval(DataFrame.label[index]),dtype='float32'))
            yield np.array(IMG),np.array(LABEL)

    # 普通数据增强模式
    elif exptype==2:
        #将这个生成器类当成图像图像处理器来用
        datagen = ImageDataGenerator(rotation_range=10,
                                    width_shift_range=0.15,
                                    height_shift_range=0.15,
                                    shear_range=0.15,
                                    zoom_range = 0.15,
                                    horizontal_flip=True)
        while True:
            IMG = []
            LABEL = []
            for _ in range(batch_size):
                im = next(zipped)[0]
                index = DataFrame.index[(DataFrame.pic_path==im)].tolist()[0]  #索引值
                img = cv2.resize(cv2.imread(im), (input_width, input_height))
                if incl_preimg:
                    IMG.append(img/ 127.5 - 1)
                    LABEL.append(np.array(eval(DataFrame.label[index]),dtype='float32'))
                if Aug:  #即增强参数存在或不等于0
                    aimg = img.reshape((1,) + img.shape)
                    data_iter = datagen.flow(aimg, batch_size=4,)  #此batch_size非彼batch_size
                    for i in range(Aug):            #只是为了循环Aug次，不需要取到i值
                        IMG.append(data_iter.next()[0]/ 127.5 - 1)  #默认float32格式
                        LABEL.append(np.array(eval(DataFrame.label[index]),dtype='float32'))
            yield np.array(IMG),np.array(LABEL)

    # 掩膜数据增强模式
    elif exptype==3:
        while True:
            IMG = []
            LABEL = []
            for _ in range(batch_size):
                im = next(zipped)[0]
                index = DataFrame.index[(DataFrame.pic_path==im)].tolist()[0]
                img = cv2.resize(cv2.imread(im), (input_width, input_height))
                
                #要能背景替换，要存在掩码；对于现代复试没有掩码的情况就不执行，即使在训练集中
                #-1值代表无掩膜文件（train）
                if (Aug!=0)&(DataFrame.mask_path[index]!=-1):  
                    bks = random.sample(comple_listdir("backgroud/"), Aug)
                    msk = cv2.resize(cv2.imread(DataFrame.mask_path[index]), (input_width, input_height))
                    revmsk=msk.copy()
                    revmsk.dtype = 'bool_'
                    revmsk = ~revmsk
                    revmsk.dtype = 'uint8'
                    for bk in bks:
                        bkp = cv2.resize(cv2.imread(bk), (input_width, input_height))
                        IMG.append((img*msk + bkp*revmsk)/ 127.5 - 1)
                        LABEL.append(np.array(eval(DataFrame.label[index]),dtype='float32'))
                if incl_preimg:
                    IMG.append(img/ 127.5 - 1)
                    LABEL.append(np.array(eval(DataFrame.label[index]),dtype='float32'))
            yield np.array(IMG),np.array(LABEL)

    elif exptype==4: # synthesize,综合
        datagen = ImageDataGenerator(rotation_range=10,
                            width_shift_range=0.15,
                            height_shift_range=0.15,
                            shear_range=0.15,
                            zoom_range = 0.15,
                            horizontal_flip=True)
        while True:
            IMG = []
            LABEL = []
            for _ in range(batch_size):
                im = next(zipped)[0]
                index = DataFrame.index[(DataFrame.pic_path==im)].tolist()[0]
                img = cv2.resize(cv2.imread(im), (input_width, input_height))
                
                #要能背景替换，要存在掩码；对于现代复试没有掩码的情况就不执行，即使在训练集中
                #-1值代表无掩膜文件（train）
                if (Aug!=0)&(DataFrame.mask_path[index]!=-1):  
                    bks = random.sample(comple_listdir("backgroud/"), Aug)
                    msk = cv2.resize(cv2.imread(DataFrame.mask_path[index]), (input_width, input_height))
                    revmsk=msk.copy()
                    revmsk.dtype = 'bool_'
                    revmsk = ~revmsk
                    revmsk.dtype = 'uint8'
                    for bk in bks:
                        bkp = cv2.resize(cv2.imread(bk), (input_width, input_height))
                        aimg = img*msk + bkp*revmsk
                        aimg = aimg.reshape((1,) + aimg.shape)
                        data_iter = datagen.flow(aimg, batch_size=1,)

                        IMG.append(data_iter.next()[0]/ 127.5 - 1)
                        
                        LABEL.append(np.array(eval(DataFrame.label[index]),dtype='float32'))
                if incl_preimg:
                    IMG.append(img/ 127.5 - 1)
                    LABEL.append(np.array(eval(DataFrame.label[index]),dtype='float32'))
            yield np.array(IMG),np.array(LABEL)



# 这个生成器是针对没有CSV文件的时候
def myDataGenerator(sub_data_path, batch_size, input_height, input_width, n_classes):
    categoy = comple_listdir(sub_data_path)  # 打开训练集或验证集
    images_list = []
    for i in range(len(categoy)):  # 文件夹循环
        for j in comple_listdir(categoy[i]):
            # 种类文件夹名称，类文件夹编号，每个图像对应的具体路径
            images_list.append((j))
    random.shuffle(images_list)
    zipped = itertools.cycle(zip(images_list))

    while True:
        IMG = []
        LABEL = []
        for _ in range(batch_size):
            im = next(zipped)[0]
            IMG.append(getImageArr(im, input_width, input_height))
            LABEL.append(to_categorical(int(im.split('/')[-2].split('_')[0]),n_classes)) # get/../`0`_miao_type1_74.jpg->'0'
        yield np.array(IMG),np.array(LABEL)


# 这个迭代器局限性很大，只能完成猫狗识别这样的任务，一旦标签复杂化，数据管理智能化（如用CSV、text、数据库），就完全不能用了
def  G1G2(input_shape,batch_size,batch_size_val):
    # 数据生成迭代器  选用自定义生成器或官方生成器，二选一
    train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    G1 = train_datagen.flow_from_directory(
        directory = './data/train/',
        target_size = input_shape[:-1],
        color_mode = 'rgb',
        classes = None,
        class_mode = 'categorical',
        batch_size = batch_size,
        shuffle = True)

    G2 = test_datagen.flow_from_directory(
        directory = './data/val/',
        target_size = input_shape[:-1],
        batch_size = batch_size_val,
        class_mode = 'categorical')

    return G1,G2

def getImageArr(path, width, height, imgNorm="1", bk=None, mask=None):
    try:
        img = cv2.imread(path)
        img = cv2.resize(img, (width, height))
        if (bk!=None)&(mask!=-1):  # 启动合成背景模式，-1值代表无掩膜文件
            bkp = cv2.resize(cv2.imread(bk), (width, height))
            msk = cv2.resize(cv2.imread(mask), (width, height))
            img = img*msk
            #掩膜01互换，得背景
            msk.dtype = 'bool_'
            msk = ~msk
            msk.dtype = 'uint8'
            img = img+bkp*msk
        if imgNorm == "1":
            img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
            return img
        if imgNorm == "2":
            img = np.float32(cv2.resize(img, (width, height))) / 255
            return img
    except Exception as e:
        print(path, e)
        img = np.zeros((height, width, 3))
        return img

'''
模型定义部分
'''

# 模型定义
def get_VGG(base,input_shape,out_node):
    if base==1:
        base_model = VGG16(input_shape = input_shape,
                        include_top = False,
                        weights = 'imagenet',
                            )
    elif base==0:
        base_model = VGG16(input_shape = input_shape,
                include_top = False,
                weights = None,
                    )
    headModel = base_model.output
    headModel = Flatten(name = 'flatten')(headModel)
    headModel = Dense(512,activation = 'relu')(headModel)
    headModel = Dense(512,activation = 'relu')(headModel)
    headModel = Dense(out_node,activation = 'softmax')(headModel)
    model = Model(inputs=base_model.input, outputs=headModel)
    # for layer in model.layers[:freeze]:  # 在incl_fc=2的情况下，freeze一般取20
    #     layer.trainable = False
    return model

def get_Xception(input_shape,out_node):
    base_model = Xception(weights='imagenet',
                        include_top=False,
                        input_shape=input_shape
                        )

    # 添加全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # 添加一个全连接层
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    # 添加一个分类器，假设我们有10个类
    predictions = Dense(out_node, activation='softmax')(x)

    # 构建我们需要训练的完整模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # 首先，我们只训练顶部的几层（随机初始化的层）
    # 锁住所有 Xception 的卷积层
    # for layer in base_model.layers:
    #     layer.trainable = False
    return model


# 胶囊网络


"""
压缩函数,我们使用0.5替代hinton论文中的1,如果是1，所有的向量的范数都将被缩小。
如果是0.5，小于0.5的范数将缩小，大于0.5的将被放大
"""
# def squash(x, axis=-1):
#     s_quared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
#     scale = K.sqrt(s_quared_norm) / (0.5 + s_quared_norm)
#     result = scale * x
#     return result


# # 定义我们自己的softmax函数，而不是K.softmax.因为K.softmax不能指定轴
# def softmax(x, axis=-1):
#     ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
#     result = ex / K.sum(ex, axis=axis, keepdims=True)
#     return result


# # 定义边缘损失，输入y_true, p_pred，返回分数，传入即可fit时候即可
# def margin_loss(y_true, y_pred):
#     lamb, margin = 0.5, 0.1
#     result = K.sum(y_true * K.square(K.relu(1 - margin -y_pred))
#     + lamb * (1-y_true) * K.square(K.relu(y_pred - margin)), axis=-1)
#     return result


# class Capsule(Layer):
#     """编写自己的Keras层需要重写3个方法以及初始化方法
#     1.build(input_shape):这是你定义权重的地方。
#     这个方法必须设self.built = True，可以通过调用super([Layer], self).build()完成。
#     2.call(x):这里是编写层的功能逻辑的地方。
#     你只需要关注传入call的第一个参数：输入张量，除非你希望你的层支持masking。
#     3.compute_output_shape(input_shape):
#      如果你的层更改了输入张量的形状，你应该在这里定义形状变化的逻辑，这让Keras能够自动推断各层的形状。
#     4.初始化方法,你的神经层需要接受的参数
#     """
#     def __init__(self,
#                  num_capsule,
#                  dim_capsule,
#                  routings=3,
#                  share_weights=True,
#                  activation='squash',
#                  **kwargs):
#         super(Capsule, self).__init__(**kwargs)  # Capsule继承**kwargs参数
#         self.num_capsule = num_capsule
#         self.dim_capsule = dim_capsule
#         self.routings = routings
#         self.share_weights = share_weights
#         if activation == 'squash':
#             self.activation = squash
#         else:
#             self.activation = activation.get(activation)  # 得到激活函数

#     # 定义权重
#     def build(self, input_shape):
#         input_dim_capsule = input_shape[-1]
#         if self.share_weights:
#             # 自定义权重
#             self.kernel = self.add_weight(
#                 name='capsule_kernel',
#                 shape=(1, input_dim_capsule,
#                        self.num_capsule * self.dim_capsule),
#                 initializer='glorot_uniform',
#                 trainable=True)
#         else:
#             input_num_capsule = input_shape[-2]
#             self.kernel = self.add_weight(
#                 name='capsule_kernel',
#                 shape=(input_num_capsule, input_dim_capsule,
#                        self.num_capsule * self.dim_capsule),
#                 initializer='glorot_uniform',
#                 trainable=True)
#         super(Capsule, self).build(input_shape)  # 必须继承Layer的build方法

#     # 层的功能逻辑(核心)
#     def call(self, inputs):
#         if self.share_weights:
#             hat_inputs = K.conv1d(inputs, self.kernel)
#         else:
#             hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

#         batch_size = K.shape(inputs)[0]
#         input_num_capsule = K.shape(inputs)[1]
#         hat_inputs = K.reshape(hat_inputs,
#                                (batch_size, input_num_capsule,
#                                 self.num_capsule, self.dim_capsule))
#         hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

#         b = K.zeros_like(hat_inputs[:, :, :, 0])
#         for i in range(self.routings):
#             c = softmax(b, 1)
#             o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
#             if K.backend() == 'theano':
#                 o = K.sum(o, axis=1)
#             if i < self.routings-1:
#                 b += K.batch_dot(o, hat_inputs, [2, 3])
#                 if K.backend() == 'theano':
#                     o = K.sum(o, axis=1)
#         return o

#     def compute_output_shape(self, input_shape):  # 自动推断shape
#         return (None, self.num_capsule, self.dim_capsule)


# def MODEL(base):
#     # input_image = Input(shape=(32, 32, 3))
#     # x = Conv2D(64, (3, 3), activation='relu')(input_image)
#     # x = Conv2D(64, (3, 3), activation='relu')(x)
#     # x = AveragePooling2D((2, 2))(x)
#     # x = Conv2D(128, (3, 3), activation='relu')(x)
#     # x = Conv2D(128, (3, 3), activation='relu')(x)

#     # 从VGG模型截取特征提取器
#     input_model = load_model(base)
#     input_model_BasePart = Model(inputs=input_model.inputs, outputs=input_model.get_layer(index=18).output)
#     x = input_model_BasePart.output
#     """
#     现在我们将它转换为(batch_size, input_num_capsule, input_dim_capsule)，然后连接一个胶囊神经层。模型的最后输出是10个维度为16的胶囊网络的长度
#     """
#     x = Reshape((-1, 128))(x)  # (None, 100, 128) 相当于前一层胶囊(None, input_num, input_dim)
#     capsule = Capsule(num_capsule=10, dim_capsule=16, routings=3, share_weights=True)(x)  # capsule-（None,10, 16)
#     output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), axis=2)))(capsule)  # 最后输出变成了10个概率值
#     model = Model(inputs=input_model_BasePart.input, output=output)
#     return model
