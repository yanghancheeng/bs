# %%
'''适配机房服务器'''
import platform
import sys
import os
sysEn = platform.system()
if sysEn == "Linux":
    print('OS is linux!!!\n')
    if os.path.exists('/usr/lib/python2.7/dist-packages'):
        sys.path.remove('/usr/lib/python2.7/dist-packages')
else:
    print('OS is windows!!!\n')
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import pandas as pd
import numpy as np
import math
import pickle
import time
import tensorflow as tf

if tf.__version__ == "1.13.1":  # 这是杨汉城的个人电脑
    acc_value = "acc"
elif tf.__version__ == "2.1.0":  # 服务器
    acc_value = "accuracy"
else:
    print("若发生acc还是accuracy选择问题，请在define中修改acc_value值")
    acc_value = '?'  # 按照自己的来


def createPath(path):
    if (not (os.path.exists(os.getcwd() + path))):
        os.mkdir(os.getcwd() + path)
        print("创建路径" + os.getcwd() + path)


createPath("\\log")
createPath("\\log\\model_save")
createPath("\\log\\plt")
createPath("\\log\\tfbd")

input_height = 224
input_width = 224
nChannels = 3
input_shape = (input_height, input_width, nChannels)


def continue_train_def(experiment, filepath, ):
    continue_train = get_eval("是否接上次模型", 1)
    pkl_path = './log/plt/{}.pkl'.format(experiment)
    if continue_train == 1 and os.access(filepath, os.F_OK):  #
        if os.access(pkl_path, os.F_OK):  # pkl文件存在。衔接训练old_tra_loss等必须存在，若是直接移值h5,则plt文件夹中是没有pkl文件的，会报错
            print("pkl文件存在")
            time.sleep(2)
            with open('./log/plt/{}.pkl'.format(experiment), 'rb') as file_pi:  # 读取上一次训练历史，以衔接完整损失精度曲线
                old_history = pickle.load(file_pi)
            old_tra_loss = old_history['loss']
            old_val_loss = old_history['val_loss']
            old_tra_acc = old_history[acc_value]  # 为防止因版本导致“acc”和“accuracy”改来改去，此变量统一在define.py中定义为acc_value
            old_val_acc = old_history['val_' + acc_value]
        else:
            print("无pkl文件！")
            old_tra_loss = []
            old_val_loss = []
            old_tra_acc = []
            old_val_acc = []
            time.sleep(2)
    else:
        if continue_train == 1:
            print("不存在该模型!!\n")
        if os.access(pkl_path, os.F_OK):
            print("不存在pkl历史训练数据文件!!\n")
        continue_train == 0
        time.sleep(2)
    return [continue_train, [old_tra_loss, old_val_loss, old_tra_acc, old_val_acc]]


# 功能函数：路径列表函数改写os.listdir ==> comple_listdir
def comple_listdir(in_path):
    path_list = os.listdir(in_path)
    for i in range(len(path_list)):
        path_list[i] = in_path + path_list[i]
        if "." in path_list[i].split('/')[-1]:  # 要给文件夹（没有.的）加'/'符号，如果是文件就不用加
            pass  # 因为检测到文件所以什么都不用做
        else:
            path_list[i] = path_list[i] + '/'
    return path_list


# 数据集训练验证分割，训练样本平衡
def get_train_val_df(file, frac=0.9, random_state=0, clothes='only_mz'):
    df = pd.read_csv(file)
    if clothes == 'only_mz':
        df = df[df.clothes == 'minzu']
    test_df = df[df.type == 'test']
    df = df[df.type == 'train']
    tra_df = df.sample(frac=frac, random_state=random_state, axis=0)  # 训练集表格
    val_df = df[~df.index.isin(tra_df.index)]  # 验证集表格
    # 因为朝鲜族只有161张，其它类别大概三四百张，直接复制一倍，用于样本平衡
    # ignore_index=True，避免index重复而产生的bug
    tra_df = tra_df.append(tra_df[tra_df.category == 'chaoxian'], ignore_index=True)
    return tra_df, val_df, test_df


# 默认值提示修改工具
def get_eval(name, default_val):
    theVal = default_val
    print(name, ' 默认:', theVal)
    temp = input("请修改输入值，或回车 ：")
    if temp != '':  # ''为回车
        theVal = eval(temp)
    print('现', name, '为：', theVal, "\n")
    return theVal


def get_str(name, default_val):
    theStr = default_val
    print(name, ' 默认:', theStr)
    temp = input("请修改输入值，或回车 ：")
    if temp != '':  # ''为回车
        theStr = temp
    print('现', name, '为：', theStr, "\n")
    return theStr


# 日志可视化，训练过程中的损失、精度变化
def training_vis(hist, result_dir, prefix):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history[acc_value]
    val_acc = hist.history['val_' + acc_value]

    # make a figure
    fig = plt.figure(figsize=(8, 4), dpi=120)
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, marker='o', label='train_loss')
    ax1.plot(val_loss, marker='o', label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss on Training and Validation Data')
    ax1.grid(ls='--')
    ax1.legend()  # 增加图例
    # plt.xlim(0)

    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc, marker='o', label='train_acc')
    ax2.plot(val_acc, marker='o', label='val_acc')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy  on Training and Validation Data')
    ax2.grid(ls='--')
    ax2.legend(loc=4)
    fig.tight_layout()
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))  # y轴刻度间距0.1
    plt.ylim(0, 1.05)  # y轴范围0到1
    #     plt.xlim(0)
    fig.savefig(os.path.join(result_dir, '{}.png'.format(prefix)))
    # plt.show()
    plt.close()


def save_history(history, result_dir, prefix):
    loss = history.history['loss']
    acc = history.history[acc_value]
    val_loss = history.history['val_loss']
    val_acc = history.history['val_' + acc_value]
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, '{}_result.txt'.format(prefix)), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


# 预测、结果测试用
def test_info(predicts, val_path):
    maxv = []
    Loc = []
    for i in predicts:
        maxv.append(max(i))
        Loc.append(np.argmax(i))
    images = os.listdir(val_path)
    images.sort()
    for i in range(len(images)):
        images[i] = val_path + images[i]  # 获得完整图像路径，而非单个的图像名
    LABEL = []
    for im in images:
        LABEL.append(int(im.split('/')[-1].split(".")[0]))
    count = 0
    for j in range(len(Loc)):
        if Loc[j] != LABEL[j]:
            count = count + 1
    print("预测错误个数：", count, '\n',
          "占总数比例：", (count / len(images)) * 100, '%\n',
          "亦即正确率：", (1 - count / len(images)) * 100, '%')


# ============== 计数数据量函数 ==============
Cnk = lambda n, k: int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))  # 组合数计算


def get_SIAMESE_SAME_NUMB(sub_data_path):
    counter = 0
    for d in os.listdir(sub_data_path):
        counter += Cnk(len(os.listdir(sub_data_path + d)), 2)
    return counter


def get_SIAMESE_DIFF_NUMB(sub_data_path, SAME_NUMB, CUT_RATIO):
    if CUT_RATIO == None:
        counter = 0
        c = []
        for d in os.listdir(sub_data_path):
            c.append(len(os.listdir(sub_data_path + d)))
        for i in range(len(c)):
            for j in range(i):
                a = c[i] * c[j]
                counter += a
        return counter
    else:
        return SAME_NUMB * CUT_RATIO


# 计算sub_data_path下所有样本量
def get_NUMB(sub_data_path):
    counter = 0
    for d in os.listdir(sub_data_path):
        counter += len(os.listdir(sub_data_path + d))
    return counter


# ============== 计数数据量函数 ==============

# 将列表元素转为字符串型的字符串拼接函数
def trans_str(info_list):
    s = ''
    for i in info_list:
        if str(type(i)) != "<class 'str'>":
            i = str(i)
        s += i
    return s
