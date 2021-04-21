# coding: utf-8
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from define import *
from my_Generator_Model import getModel, myAugDataGenerator
import pickle
import os
import cv2


def display():
    print('该次实验为：', experiment,
          '\n可能的图像增强模式:', expType, '\n',
          '\n训练好的模型将会保存在:', filepath, '\n',
          '有以下类别需要训练：', os.listdir('data/train/'), '\n',  # 最好再说一下具体的数目
          '\n训练集目录数据量:  ', Train_Num,
          '\n验证集目录数据量:  ', Val_Num,
          '\n测试集目录数据量:  ', Test_Num,
          '\n图像高度input_height：', input_width,
          '\n图像宽度input_width: ', input_height,
          '\n通道数nChannels: ', nChannels, '通道',
          '\n参与训练类别: ', len(traDf['category'].value_counts()), '类', '\n')
    stop = input("是否继续? [Y]/N ：")
    if ((stop == 'n') or (stop == 'N')):
        exit()

# experiment_dir
edr = {
    # 实验名[0], 实验类型[1]（关系到数据生成方法）, 学习率[2], 训练是否包含原图[3] ,训练集验证集比值[4],数据集划分种子[5],学习率衰减系数[6]
    # 实验类型对应。1：生成器无任何增强  2：普通数据增强  3：掩膜背景增强  4：混合（2和3）增强
    #   [0]       [1]   [2]    [3] [4]   [5] [6]
    0:['none',     1,   0.005, 1,  0.9,  0,  0.001, ],
    1:['ImageNet', 1,   1e-5,  1,  0.9,  0,  0.001, ],  # 训练集与验证集的处理方式一样，不进行普通数据增强也不进行背景增强 学习率起始Start 1e-5
    2:['aug',      2,   1e-5,  0,  0.9,  0,  0.001, ],  # start 1e-5  前12轮左右不进行任何增强，即aug手动置0，12轮后可看到收敛，再开启增强
    3:['mask',     3,   1e-5,  0,  0.9,  0,  0.001, ],  # 增强开启同上，在12轮后
    4:['mask_aug', 4,   1e-5,  0,  0.9,  0,  0.001, ],  # 同上
    5:['Xcep',     4,   2e-4,  0,  0.9,  0,  0.001, ],  # 可能为最优实验，增强开启时间由于实验做得少暂不确定
}

if __name__ == '__main__':

    print(edr)
    sel = get_eval("实验", 5)
    exp = edr[sel]
    experiment = exp[0]
    expType = exp[1]
    filepath = './log/model_save/' + experiment + '.h5'  # 模型保存路径
    # 学习率
    lr = get_eval("学习率", exp[2])
    print("学习率衰减系数", exp[6], '\n')
    # 数据集
    dFile = 'dataset.csv'
    print(dFile)
    traDf, valDf, testDf = get_train_val_df(dFile)

    # 增强倍数
    if expType == 1:
        Aug = 0
        inclPreImg = 1
    else:
        Aug = get_eval("增强倍数Aug", 6)
        if Aug:
            inclPreImg = get_eval("inclPreImg", exp[3])  # train_include_preimage
        else:
            inclPreImg = 1  # 如果不增强，那么训练的就只有原图，此时原图必须存在

    epochs = get_eval("epochs", 1)
    batch_size_val = batch_size = get_eval("batch_size", 4)

    Train_Num = traDf.shape[0]  # 训练集数据量（整型）
    Val_Num = valDf.shape[0]  # 验证集数据量（整型）
    Test_Num = testDf.shape[0]  # 验证集数据量（整型）
    display()

    # 数据集迭代器
    print('expType:', expType)
    G1 = myAugDataGenerator(traDf, expType, Aug, batch_size, input_height, input_width, incl_preimg=inclPreImg)
    G2 = myAugDataGenerator(valDf, 1, 0, batch_size, input_height, input_width, incl_preimg=1)
    GTest = myAugDataGenerator(testDf, 1, 0, 2, input_height, input_width, incl_preimg=1)
    # 训练集生成抽样
    a = next(G1)
    cv2.imwrite('ck0.png', (a[0][0] + 1) * 127.5)
    cv2.imwrite('ck1.png', (a[0][1] + 1) * 127.5)

    # 模型获取
    continue_train = continue_train_def(experiment, filepath)  # 续训检测
    model = getModel(sel, continue_train, input_shape, filepath)

    '''编译参数选择——交叉熵损失或二元损失'''
    model.compile(loss='categorical_crossentropy',  # binary_crossentropy categorical_crossentropy
                  optimizer=SGD(lr=lr, decay=exp[6], momentum=0.9, nesterov=True),
                  # SGD(lr=1e-4,momentum=0.9)  Adam(lr=1e-4) , decay=1e-7
                  metrics=['accuracy'])  # ,get_lr_metric(optimizer)
    model.summary()

    # 回调
    # checkpoint = ModelCheckpoint(filepath, monitor='val_'+acc_value, verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False,
                                 mode='auto')

    callbacks_list = [checkpoint,
                      # tensorflow.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),  # Interrupt training if `val_loss` stops improving for over 2 epochs
                      ]


    def model_train():  # 因为作用域问题，该函数只能定义在这里
        the_history = model.fit_generator(G1,
                                          Train_Num // batch_size + 1,
                                          callbacks=callbacks_list,
                                          validation_data=G2,
                                          validation_steps=Val_Num // batch_size + 1,  # 因为验证集不必要增强运算和内存占用，能省出更多资源利用
                                          epochs=epochs, )

        score = model.evaluate_generator(GTest, Test_Num // 2)
        print("样本准确率%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

        if continue_train == 1:
            the_history.history['loss'] = old_tra_loss + the_history.history['loss']
            the_history.history['val_loss'] = old_val_loss + the_history.history['val_loss']
            the_history.history[acc_value] = old_tra_acc + the_history.history[acc_value]
            the_history.history['val_' + acc_value] = old_val_acc + the_history.history['val_' + acc_value]
        training_vis(the_history, './log/plt/', experiment)  # experiment为所作实验名
        save_history(the_history, './log/plt/', experiment)
        # 保存

        with open('./log/plt/{}.pkl'.format(experiment), 'wb') as file_pi:
            pickle.dump(the_history.history, file_pi)

        return the_history

    history = model_train()

    while True:  # 如果效果不好,又不想麻烦地重新执行程序再训，那么这里通过循环判断是否需要再训
        try:
            print("实验学习率为", lr)
            epochs = int(input("如果不训，请敲回车，如果还训，请输入轮数："))
        except Exception as ex:
            print("再好好想想！")
            try:
                epochs = int(input("真的不训了吗，如果反悔了，请输入轮数："))
            except Exception as ex:
                print("再好好想想，程序一旦退出，再从这里训练就很麻烦！")
                try:
                    epochs = int(input("请输入本次轮数："))
                except:
                    print("程序结束~")
                    exit()
        else:
            continue_train = 1
            old_tra_loss = history.history['loss']
            old_val_loss = history.history['val_loss']
            old_tra_acc = history.history[acc_value]  # 为防止因版本导致“acc”和“accuracy”改来改去，此变量统一在define.py中定义为acc_value
            old_val_acc = history.history['val_' + acc_value]

            history = model_train()
