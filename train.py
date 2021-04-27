# coding: utf-8
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint  # EarlyStopping
from define import *
from my_Generator_Model import getModel, myAugDataGenerator
import pickle
from copy import deepcopy
from hxConfMat import myCMPlot


def display():
    print('该次实验为：', experiment,
          '\n可能的图像增强模式:', exp['typ'], '\n',
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
    if (stop == 'n') or (stop == 'N'):
        exit()


# experiment_dir
edr = {
    # 实验名[0], 实验类型[1]（关系到数据生成方法）, 学习率[2], 训练是否包含原图[3] ,训练集验证集比值[4],数据集划分种子[5],学习率衰减系数[6]
    # 实验类型对应。1：生成器无任何增强  2：普通数据增强  3：掩膜背景增强  4：混合（2和3）增强
    0: {'exp': 'none',     'typ': 1, 'lr': 5e-3, 'inPre': 1, 'TVSpl': 0.9, 'sed': 0, 'decay': 0.001, 'Aug': 0},
    1: {'exp': 'ImageNet', 'typ': 1, 'lr': 1e-5, 'inPre': 1, 'TVSpl': 0.9, 'sed': 0, 'decay': 0.001, 'Aug': 0},
    2: {'exp': 'aug',      'typ': 2, 'lr': 1e-5, 'inPre': 0, 'TVSpl': 0.9, 'sed': 0, 'decay': 0.001, 'Aug': 4},
    3: {'exp': 'mask',     'typ': 3, 'lr': 1e-5, 'inPre': 0, 'TVSpl': 0.9, 'sed': 0, 'decay': 0.001, 'Aug': 4},
    4: {'exp': 'mask_aug', 'typ': 4, 'lr': 1e-5, 'inPre': 0, 'TVSpl': 0.9, 'sed': 0, 'decay': 0.001, 'Aug': 4},
    5: {'exp': 'Xcep',     'typ': 4, 'lr': 2e-4, 'inPre': 0, 'TVSpl': 0.9, 'sed': 0, 'decay': 0.001, 'Aug': 4},
    6: {'exp': 'Dense',    'typ': 4, 'lr': 2e-4, 'inPre': 0, 'TVSpl': 0.9, 'sed': 0, 'decay': 0.001, 'Aug': 4},
}

if __name__ == '__main__':
    print(pd.DataFrame(edr).T)
    sel = get_eval("实验", 6)
    exp = edr[sel]  # dic
    experiment = exp['exp']
    filepath = './log/model_save/' + experiment + '.h5'  # 模型保存路径
    # 学习率
    exp['lr'] = get_eval("学习率", exp['lr'])
    print("学习率衰减系数", exp['decay'], '\n')
    # 数据集
    traDf, valDf, testDf = get_train_val_df('dataset.csv')

    # 增强倍数
    if exp['typ'] != 1:  # 非类型一必要定义增强倍数
        exp['Aug'] = get_eval("增强倍数Aug", exp['Aug'])
        exp['inPre'] = get_eval("inclPreImg", exp['inPre'])  # train_include_preimage

    batch_size_val = batch_size = get_eval("batch_size", 4)
    epochs = get_eval("epochs", 1)

    Train_Num = traDf.shape[0]  # 训练集数据量（整型）
    Val_Num = valDf.shape[0]  # 验证集数据量（整型）
    Test_Num = testDf.shape[0]  # 验证集数据量（整型）
    display()

    # 数据集迭代器
    print('生成器类型:', exp['typ'])
    G1 = myAugDataGenerator(traDf, exp['typ'], exp['Aug'], batch_size, input_height, input_width,
                            incl_preimg=exp['inPre'])
    G2 = myAugDataGenerator(valDf, 1, 0, batch_size, input_height, input_width, incl_preimg=1)
    GTest = myAugDataGenerator(testDf, 1, 0, 2, input_height, input_width, incl_preimg=1)
    # 训练集生成抽样
    # a = next(G1)
    # cv2.imwrite('ck0.png', (a[0][0] + 1) * 127.5)
    # cv2.imwrite('ck1.png', (a[0][1] + 1) * 127.5)

    # 续训与模型获取
    continue_train, oldHistoryList = continue_train_def(experiment, filepath)  # 续训检测
    model = getModel(sel, continue_train, input_shape, filepath)

    '''编译参数选择——交叉熵损失或二元损失'''
    model.compile(loss='categorical_crossentropy',  # binary_crossentropy categorical_crossentropy
                  optimizer=SGD(lr=exp['lr'], decay=exp['decay'], momentum=0.9, nesterov=True),
                  # SGD(lr=1e-4,momentum=0.9)  Adam(lr=1e-4) , decay=1e-7
                  metrics=['accuracy'])  # ,get_lr_metric(optimizer)
    model.summary()

    # 回调 checkpoint = ModelCheckpoint(filepath, monitor='val_'+acc_value, verbose=1, save_best_only=True,
    # save_weights_only=False, mode='max')
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False,
                                 mode='auto')

    callbacks_list = [checkpoint,
                      # tensorflow.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),  # Interrupt
                      # training if `val_loss` stops improving for over 2 epochs
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
        if_CM = get_eval("是否绘制混淆矩阵", 0)
        if if_CM is 1:
            myCMPlot(model, experiment)
        if continue_train == 1:
            the_history.history['loss'] = oldHistoryList[0] + the_history.history['loss']
            the_history.history['val_loss'] = oldHistoryList[1] + the_history.history['val_loss']
            the_history.history[acc_value] = oldHistoryList[2] + the_history.history[acc_value]
            the_history.history['val_' + acc_value] = oldHistoryList[3] + the_history.history['val_' + acc_value]
        training_vis(the_history, './log/plt/', experiment)  # experiment为所作实验名
        save_history(the_history, './log/plt/', experiment)
        # 保存

        with open('./log/plt/{}.pkl'.format(experiment), 'wb') as file_pi:
            pickle.dump(the_history.history, file_pi)

        return the_history


    history = model_train()

    while True:  # 如果效果不好,又不想麻烦地重新执行程序再训，那么这里通过循环判断是否需要再训
        try:
            print("实验学习率为", exp['lr'])
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
            oldHistoryList = [deepcopy(history.history['loss']),
                              deepcopy(history.history['val_loss']),
                              deepcopy(history.history[acc_value]),
                              deepcopy(history.history['val_' + acc_value])]

            history = model_train()
