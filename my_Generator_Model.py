# from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.utils import to_categorical
from define import comple_listdir
import itertools
import cv2
import os
import pandas as pd
import numpy as np
import random
from define import get_str


def get_Dense(input_shape, out_node):
    base_model = DenseNet121(include_top=False,
                             weights='imagenet',
                             input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(out_node, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # for layer in model.layers[:freeze]:  # 在incl_fc=2的情况下，freeze一般取20
    #     layer.trainable = False
    return model


def getModel(sel, continue_train, input_shape, filepath):
    def initMode():
        if sel == 0:
            model = get_VGG(0, input_shape, out_node)
        elif sel == 1:
            model = get_VGG(1, input_shape, out_node)  # ImageNet
        elif sel == 2:
            model = get_VGG(1, input_shape, out_node)  # aug
        elif sel == 3:
            model = get_VGG(1, input_shape, out_node)  # mask
        elif sel == 4:
            model = get_VGG(1, input_shape, out_node)  # mask_aug
        elif sel == 5:
            model = get_Xception(input_shape, out_node)  # Xecp
        else:
            model = get_Dense(input_shape, out_node)
        return model

    out_node = 10

    if continue_train == 1 and os.access(filepath, os.F_OK):
        mdf = get_str('读入权重文件（完整路径）', filepath)
        return load_model(mdf)
    else:
        if continue_train == 1:
            print("接上次模型，但模型文件丢失，使用随机权重")
        return initMode()


def getImageArr(path, width, height, imgNorm="1", bk=None, mask=None):
    try:
        img = cv2.imread(path)
        img = cv2.resize(img, (width, height))
        if (bk is not None) & (mask != -1):  # 启动合成背景模式，-1值代表无掩膜文件
            bkp = cv2.resize(cv2.imread(bk), (width, height))
            msk = cv2.resize(cv2.imread(mask), (width, height))
            img = img * msk
            # 掩膜01互换，得背景
            msk.dtype = 'bool_'
            msk = ~msk
            msk.dtype = 'uint8'
            img = img + bkp * msk
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


# 貌似用不到getImageArr
def myAugDataGenerator(DataFrame, exptype, Aug, batch_size, input_height, input_width, incl_preimg=True):
    DataFrame = DataFrame.sample(frac=1)  # 随机打乱顺序
    zipped = itertools.cycle(zip(DataFrame.pic_path))

    # 无数据增强模式
    if exptype == 1:
        while True:
            IMG = []
            LABEL = []
            for _ in range(batch_size):
                im = next(zipped)[0]
                index = DataFrame.index[(DataFrame.pic_path == im)].tolist()[0]  # 索引值
                img = cv2.resize(cv2.imread(im), (input_width, input_height))
                IMG.append(img / 127.5 - 1)  # 默认float32格式
                LABEL.append(np.array(eval(DataFrame.label[index]), dtype='float32'))
            yield np.array(IMG), np.array(LABEL), [None]

    # 普通数据增强模式
    elif exptype == 2:
        # 将这个生成器类当成图像图像处理器来用
        datagen = ImageDataGenerator(rotation_range=10,
                                     width_shift_range=0.15,
                                     height_shift_range=0.15,
                                     shear_range=0.15,
                                     zoom_range=0.15,
                                     horizontal_flip=True)
        while True:
            IMG = []
            LABEL = []
            for _ in range(batch_size):
                im = next(zipped)[0]
                index = DataFrame.index[(DataFrame.pic_path == im)].tolist()[0]  # 索引值
                img = cv2.resize(cv2.imread(im), (input_width, input_height))
                if incl_preimg:
                    IMG.append(img / 127.5 - 1)
                    LABEL.append(np.array(eval(DataFrame.label[index]), dtype='float32'))
                if Aug:  # 即增强参数存在或不等于0
                    aimg = img.reshape((1,) + img.shape)
                    data_iter = datagen.flow(aimg, batch_size=4, )  # 此batch_size非彼batch_size
                    for i in range(Aug):  # 只是为了循环Aug次，不需要取到i值
                        IMG.append(data_iter.next()[0] / 127.5 - 1)  # 默认float32格式
                        LABEL.append(np.array(eval(DataFrame.label[index]), dtype='float32'))
            yield np.array(IMG), np.array(LABEL), [None]

    # 掩膜数据增强模式
    elif exptype == 3:
        while True:
            IMG = []
            LABEL = []
            for _ in range(batch_size):
                im = next(zipped)[0]
                index = DataFrame.index[(DataFrame.pic_path == im)].tolist()[0]
                img = cv2.resize(cv2.imread(im), (input_width, input_height))

                # 要能背景替换，要存在掩码；对于现代复试没有掩码的情况就不执行，即使在训练集中
                # -1值代表无掩膜文件（train）
                if (Aug != 0) & (DataFrame.mask_path[index] != -1):
                    bks = random.sample(comple_listdir("backgroud/"), Aug)
                    msk = cv2.resize(cv2.imread(DataFrame.mask_path[index]), (input_width, input_height))
                    revmsk = msk.copy()
                    revmsk.dtype = 'bool_'
                    revmsk = ~revmsk
                    revmsk.dtype = 'uint8'
                    for bk in bks:
                        bkp = cv2.resize(cv2.imread(bk), (input_width, input_height))
                        IMG.append((img * msk + bkp * revmsk) / 127.5 - 1)
                        LABEL.append(np.array(eval(DataFrame.label[index]), dtype='float32'))
                if incl_preimg:
                    IMG.append(img / 127.5 - 1)
                    LABEL.append(np.array(eval(DataFrame.label[index]), dtype='float32'))
            yield np.array(IMG), np.array(LABEL), [None]

    elif exptype == 4:  # synthesize,综合
        datagen = ImageDataGenerator(rotation_range=10,
                                     width_shift_range=0.15,
                                     height_shift_range=0.15,
                                     shear_range=0.15,
                                     zoom_range=0.15,
                                     horizontal_flip=True)
        while True:
            IMG = []
            LABEL = []
            for _ in range(batch_size):
                im = next(zipped)[0]
                index = DataFrame.index[(DataFrame.pic_path == im)].tolist()[0]
                img = cv2.resize(cv2.imread(im), (input_width, input_height))

                # 要能背景替换，要存在掩码；对于现代复试没有掩码的情况就不执行，即使在训练集中
                # -1值代表无掩膜文件（train）
                if (Aug != 0) & (DataFrame.mask_path[index] != -1):
                    bks = random.sample(comple_listdir("backgroud/"), Aug)
                    msk = cv2.resize(cv2.imread(DataFrame.mask_path[index]), (input_width, input_height))
                    revmsk = msk.copy()
                    revmsk.dtype = 'bool_'
                    revmsk = ~revmsk
                    revmsk.dtype = 'uint8'
                    for bk in bks:
                        bkp = cv2.resize(cv2.imread(bk), (input_width, input_height))
                        aimg = img * msk + bkp * revmsk
                        aimg = aimg.reshape((1,) + aimg.shape)
                        data_iter = datagen.flow(aimg, batch_size=1, )

                        IMG.append(data_iter.next()[0] / 127.5 - 1)

                        LABEL.append(np.array(eval(DataFrame.label[index]), dtype='float32'))
                if incl_preimg:
                    IMG.append(img / 127.5 - 1)
                    LABEL.append(np.array(eval(DataFrame.label[index]), dtype='float32'))
            yield np.array(IMG), np.array(LABEL), [None]


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
            LABEL.append(
                to_categorical(int(im.split('/')[-2].split('_')[0]), n_classes))  # get/../`0`_miao_type1_74.jpg->'0'
        yield np.array(IMG), np.array(LABEL), [None]


'''
模型定义部分
'''


# 模型定义
def get_VGG(base, input_shape, out_node):
    if base == 1:
        base_model = VGG16(input_shape=input_shape,
                           include_top=False,
                           weights='imagenet',
                           )
    elif base == 0:
        base_model = VGG16(input_shape=input_shape,
                           include_top=False,
                           weights=None,
                           )
    headModel = base_model.output
    headModel = Flatten(name='flatten')(headModel)
    headModel = Dense(512, activation='relu')(headModel)
    headModel = Dense(512, activation='relu')(headModel)
    headModel = Dense(out_node, activation='softmax')(headModel)
    model = Model(inputs=base_model.input, outputs=headModel)
    # for layer in model.layers[:freeze]:  # 在incl_fc=2的情况下，freeze一般取20
    #     layer.trainable = False
    return model


def get_Xception(input_shape, out_node):
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
