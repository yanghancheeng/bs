# coding: utf-8
'''适配机房服务器'''
import platform
import sys
import time

systemp = platform.system()
if systemp == "Linux":
    # sys.path.remove('/usr/lib/python2.7/dist-packages')
    print('OS is linux!!!\n')
else:
    print('OS is windows!!!\n')

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from define import *
from my_Generator_Model import getModel,getImageArr,myDataGenerator,myAugDataGenerator,G1G2,get_VGG,get_Xception
import pickle
import os
import cv2

def display():
    print('该次实验为：',experiment,
        '\n可能的图像增强模式:',exptype,'\n',
        '\n训练好的模型将会保存在:',filepath,'\n',
        '有以下类别需要训练：',os.listdir('data/train/'),'\n',  # 最好再说一下具体的数目
        '\n训练集目录数据量:  ',Train_Num,
        '\n验证集目录数据量:  ',Val_Num,
        '\n测试集目录数据量:  ',Test_Num,
        '\n图像高度input_height：',input_width,
        '\n图像宽度input_width: ',input_height,
        '\n通道数nChannels: ',nChannels,'通道',
        '\n参与训练类别: ',len(tra_df['category'].value_counts()),'类','\n')
    stop = input("是否继续? [Y]/N ：")
    if ((stop=='n')or(stop=='N')):
        exit()
        
# experiment_dir
edr = {
    # 实验名[0], 实验类型[1]（关系到数据生成方法）, 学习率[2], 训练是否包含原图[3] ,训练集验证集比值[4],数据集划分种子[5],学习率衰减系数[6]
    # 实验类型对应。1：生成器无任何增强  2：普通数据增强  3：掩膜背景增强  4：混合（2和3）增强
    #   [0]       [1]   [2]    [3] [4]   [5] [6]
    0:['none',     1,   0.005, 1,  0.9,  0,  0.001,],
    1:['ImageNet', 1,   1e-5,  1,  0.9,  0,  0.001,],  #训练集与验证集的处理方式一样，不进行普通数据增强也不进行背景增强 学习率起始strat 1e-5
    2:['aug',      2,   1e-5,  0,  0.9,  0,  0.001,],  #strat 1e-5  前12轮左右不进行任何增强，即aug手动置0，12轮后可看到收敛，再开启增强
    3:['mask',     3,   1e-5,  0,  0.9,  0,  0.001,],  #增强开启同上，在12轮后
    4:['mask_aug', 4,   1e-5,  0,  0.9,  0,  0.001,],  #同上
    5:['Xcep',     4,   2e-4,  0,  0.9,  0,  0.001,],  #可能为最优实验，增强开启时间由于实验做得少暂不确定
}

if __name__ == '__main__':

    print(edr)
    sel = get_eval("实验",5)
    exp = edr[sel]
    experiment = exp[0]
    exptype = exp[1]
    filepath='./log/model_save/'+experiment+'.h5'# 模型保存路径
    # 学习率
    lr = get_eval("学习率",exp[2])
    print("学习率衰减系数",exp[6],'\n')
    #数据集
    dfile = 'dataset.csv'
    print(dfile)
    tra_df,val_df,test_df = get_train_val_df(dfile)

    #增强倍数
    if exptype==1:
        Aug = 0
        incl_preimg = 1
    else:
        Aug = get_eval("增强倍数Aug",6)
        if Aug:
            incl_preimg = get_eval("incl_preimg",exp[3])  # train_include_preimage
        else:
            incl_preimg = 1  #如果不增强，那么训练的就只有原图，此时原图必须存在

    
    epochs=get_eval("epochs",1)
    batch_size_val = batch_size =get_eval("batch_size",4)
    
    continue_train =get_eval("是否接上次模型",1)
    if continue_train==1 and os.access(filepath, os.F_OK):
        print("载入模型名称",filepath)
        if (not (os.path.exists('./log/plt/{}.pkl'.format(experiment)))):  #存在性判定，衔接训练old_tra_loss等必须存在，若是直接移值h5,则plt文件夹中是没有pkl文件的，会报错
            old_tra_loss = []
            old_val_loss = []
            old_tra_acc = []
            old_val_acc = []
        else:
            with open('./log/plt/{}.pkl'.format(experiment), 'rb') as file_pi:  #读取上一次训练历史，以衔接完整损失精度曲线
                old_history = pickle.load(file_pi)
            old_tra_loss = old_history['loss']
            old_val_loss = old_history['val_loss']
            old_tra_acc = old_history[acc_value]   # 为防止因版本导致“acc”和“accuracy”改来改去，此变量统一在define.py中定义为acc_value
            old_val_acc = old_history['val_'+acc_value]
    else:
        print("不存在该模型!!\n")
        time.sleep(1)
    
    Train_Num = tra_df.shape[0] # 训练集数据量（整型）
    Val_Num = val_df.shape[0]   # 验证集数据量（整型）
    Test_Num = test_df.shape[0] # 验证集数据量（整型）
    display()

    # 数据集迭代器
    print('exptype:',exptype)
    G1 = myAugDataGenerator(tra_df, exptype, Aug, batch_size,input_height, input_width, incl_preimg=incl_preimg)
    G2 = myAugDataGenerator(val_df, 1, 0, batch_size, input_height, input_width, incl_preimg=1)
    Gtest = myAugDataGenerator(test_df, 1, 0, 2,input_height, input_width, incl_preimg=1)
    # 训练集生成抽样
    a=next(G1)
    cv2.imwrite('ck0.png', (a[0][0]+1)*127.5)
    cv2.imwrite('ck1.png', (a[0][1]+1)*127.5)

    #模型获取
    model = getModel(sel,continue_train,input_shape,filepath)

    # base_model = VGG16(input_shape = input_shape,
    #         include_top = False,
    #         weights = None,
    #             )
    # headModel = base_model.output
    # headModel = Flatten(name = 'flatten')(headModel)
    # headModel = Dense(512,activation = 'relu')(headModel)
    # headModel = Dense(512,activation = 'relu')(headModel)
    # headModel = Dense(10,activation = 'softmax')(headModel)
    # model = Model(inputs=base_model.input, outputs=headModel)

    '''编译参数选择——交叉熵损失或二元损失'''
    model.compile(loss='categorical_crossentropy',  #binary_crossentropy categorical_crossentropy
            optimizer=SGD(lr=lr, decay=exp[6], momentum=0.9, nesterov=True),  #SGD(lr=1e-4,momentum=0.9)  Adam(lr=1e-4) , decay=1e-7
            metrics=['accuracy'])  #,get_lr_metric(optimizer)
    model.summary()

    # 回调
    # checkpoint = ModelCheckpoint(filepath, monitor='val_'+acc_value, verbose=1, save_best_only=True, save_weights_only=False, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
    
    callbacks_list = [checkpoint,
    # tensorflow.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),  # Interrupt training if `val_loss` stops improving for over 2 epochs
    ]

    # 训练
    history = model.fit_generator(G1, 
                                    Train_Num//batch_size+1, 
                                    callbacks=callbacks_list, 
                                    validation_data=G2, 
                                    validation_steps=Val_Num//batch_size+1, #因为验证集不必要增强运算和内存占用，能省出更多资源利用
                                    epochs=epochs,)

    score = model.evaluate_generator(Gtest,Test_Num//2)
    print("样本准确率%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
    if continue_train==1:
        history.history['loss'] = old_tra_loss + history.history['loss']
        history.history['val_loss'] = old_val_loss + history.history['val_loss']
        history.history[acc_value] = old_tra_acc + history.history[acc_value] 
        history.history['val_'+acc_value] = old_val_acc + history.history['val_'+acc_value] 
    training_vis(history, './log/plt/', experiment)  # experiment为所作实验名
    save_history(history, './log/plt/', experiment)
    #保存
    with open('./log/plt/{}.pkl'.format(experiment), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    while True:  # 如果效果不好,又不想麻烦地重新执行程序再训，那么这里通过循环判断是否需要再训
        try:
            print("实验学习率为",lr)    
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
            old_tra_loss = history.history['loss']
            old_val_loss = history.history['val_loss']
            old_tra_acc = history.history[acc_value]   # 为防止因版本导致“acc”和“accuracy”改来改去，此变量统一在define.py中定义为acc_value
            old_val_acc = history.history['val_'+acc_value]


            history = model.fit_generator(G1, 
                                            Train_Num//batch_size+1, 
                                            callbacks=callbacks_list, 
                                            validation_data=G2, 
                                            validation_steps=Val_Num//batch_size+1, #因为验证集不必要增强运算和内存占用，能省出更多资源利用
                                            epochs=epochs,)
            score = model.evaluate_generator(Gtest,Test_Num//2)
            print("样本准确率%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

            history.history['loss'] = old_tra_loss + history.history['loss']
            history.history['val_loss'] = old_val_loss + history.history['val_loss']
            history.history[acc_value] = old_tra_acc + history.history[acc_value] 
            history.history['val_'+acc_value] = old_val_acc + history.history['val_'+acc_value] 

            training_vis(history, './log/plt/', experiment)  # experiment为所作实验名
            save_history(history, './log/plt/', experiment)
            with open('./log/plt/{}.pkl'.format(experiment), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
