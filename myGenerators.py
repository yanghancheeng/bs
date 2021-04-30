from tensorflow.keras.preprocessing.image import ImageDataGenerator
from define import comple_listdir
import itertools
import cv2
import numpy as np
import random


# 外置图像处理器
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


'''
四种生成器
'''


# 2.# have tradition aug not mask aug  无任何增强，也可以用做验证集生成器
def g_NTa_NMa(DataFrame,
              Aug,
              batch_size,
              input_height=224,
              input_width=224,
              incl_preimg=True):
    DataFrame = DataFrame.sample(frac=1)  # 随机打乱顺序
    zipped = itertools.cycle(zip(DataFrame.pic_path))

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


# 2.# have tradition aug not mask aug   只传统增强
def g_Ta_NMa(DataFrame,
             Aug,
             batch_size,
             input_height=224,
             input_width=224,
             incl_preimg=True):
    DataFrame = DataFrame.sample(frac=1)  # 随机打乱顺序
    zipped = itertools.cycle(zip(DataFrame.pic_path))

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


# 3.# not tradition aug not mask aug    只掩膜增强
def g_NTa_Ma(DataFrame,
             Aug,
             batch_size,
             input_height=224,
             input_width=224,
             incl_preimg=True):
    DataFrame = DataFrame.sample(frac=1)  # 随机打乱顺序
    zipped = itertools.cycle(zip(DataFrame.pic_path))

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


# 4.# tradition aug and mask aug   双增强
def g_Ta_Ma(DataFrame,
            Aug,
            batch_size,
            input_height=224,
            input_width=224,
            incl_preimg=True):
    DataFrame = DataFrame.sample(frac=1)  # 随机打乱顺序
    zipped = itertools.cycle(zip(DataFrame.pic_path))

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


genDic = {
    "NTa_NMa": g_NTa_NMa,  # not tradition aug not mask aug  无任何增强
    "Ta_NMa": g_Ta_NMa,  # have tradition aug not mask aug   只传统增强
    "NTa_Ma": g_NTa_Ma,  # not tradition aug not mask aug    只掩膜增强
    "Ta_Ma": g_Ta_Ma,  # tradition aug and mask aug   双增强
}


# 标准生成器选取器
def myDataGenerator(genTyp, *args, **kwargs):
    return genDic[genTyp](*args, **kwargs)

# # 这个生成器是针对没有CSV文件的时候
# def myDataGenerator(sub_data_path, batch_size, input_height, input_width, n_classes):
#     categoy = comple_listdir(sub_data_path)  # 打开训练集或验证集
#     images_list = []
#     for i in range(len(categoy)):  # 文件夹循环
#         for j in comple_listdir(categoy[i]):
#             # 种类文件夹名称，类文件夹编号，每个图像对应的具体路径
#             images_list.append((j))
#     random.shuffle(images_list)
#     zipped = itertools.cycle(zip(images_list))
#
#     while True:
#         IMG = []
#         LABEL = []
#         for _ in range(batch_size):
#             im = next(zipped)[0]
#             IMG.append(getImageArr(im, input_width, input_height))
#             LABEL.append(
#                 to_categorical(int(im.split('/')[-2].split('_')[0]), n_classes))  # get/../`0`_miao_type1_74.jpg->'0'
#         yield np.array(IMG), np.array(LABEL), [None]
