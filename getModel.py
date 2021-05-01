# from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model  # , Model
import os
import time
import pickle
from define import acc_value, get_eval
from models import *
from labs import *

l_classify_model_type = l_name_vgg + \
                        l_name_vgg19 + \
                        l_name_DenseNet + \
                        l_name_DenseNet169 + \
                        l_name_DenseNet201 + \
                        l_name_Xcep + \
                        l_name_MobileNet

l_complex_model_type = l_name_MinZuNet

# 字典类型
get_model_fromDic = {}
# get_classifyModels->分类模型大类
get_model_fromDic.update({simple_model_name: get_classifyModels for simple_model_name in l_classify_model_type})


# 续训检测
# 要求返回模型，续训历史
def getModel(wight_save_path, *args, **kwargs):
    experiment = wight_save_path.split('/')[-1].split('.')[0]
    pklPath = './log/plt/{}.pkl'.format(experiment)
    if_wight = os.access(wight_save_path, os.F_OK)
    if_HisPkl = os.access(pklPath, os.F_OK)

    if if_wight or if_HisPkl:  # 权重文件和pkl有其一，就可能为续训
        continue_train = 1
    else:
        continue_train = 0  # 权重文件和pkl都没有，默认为初训
    continue_train = get_eval("是否接上次模型", continue_train)
    if continue_train:
        print('次训练~')
        if if_wight:
            print(wight_save_path, '模型文件存在')
            time.sleep(2)
            if if_HisPkl:
                print('有历史pkl文件')
                time.sleep(1)
                print('载入pkl文件', './log/plt/{}.pkl'.format(experiment))
                time.sleep(1)
                with open('./log/plt/{}.pkl'.format(experiment), 'rb') as file_pi:  # 读取上一次训练历史，以衔接完整损失精度曲线
                    old_history = pickle.load(file_pi)
                old_tra_loss = old_history['loss']
                old_val_loss = old_history['val_loss']
                old_tra_acc = old_history[acc_value]  # 为防止因版本导致“acc”和“accuracy”改来改去，此变量统一在define.py中定义为acc_value
                old_val_acc = old_history['val_' + acc_value]
                # wight_save_path = get_str('载入模型名称', wight_save_path)
                print('载入', wight_save_path)
                time.sleep(1)
                return load_model(wight_save_path), [old_tra_loss, old_val_loss, old_tra_acc, old_val_acc]
            else:
                print('但无历史pkl文件')
                time.sleep(2)
                old_tra_loss, old_val_loss, old_tra_acc, old_val_acc, = [], [], [], []
                print('载入', wight_save_path)
                time.sleep(1)
                return load_model(wight_save_path), [old_tra_loss, old_val_loss, old_tra_acc, old_val_acc]
        else:
            print("接上次模型，但模型文件丢失，使用新权重，且pkl历史归零！\n次训练转为\n初训练~")
            time.sleep(4)
            old_tra_loss, old_val_loss, old_tra_acc, old_val_acc, = [], [], [], []
            return get_model_fromDic[kwargs['model_name']](*args, **kwargs), [old_tra_loss, old_val_loss,
                                                                              old_tra_acc, old_val_acc]

    else:
        print('初训练')
        time.sleep(2)
        old_tra_loss, old_val_loss, old_tra_acc, old_val_acc, = [], [], [], []
        return get_model_fromDic[kwargs['model_name']](*args, **kwargs), [old_tra_loss, old_val_loss,
                                                                          old_tra_acc, old_val_acc]
