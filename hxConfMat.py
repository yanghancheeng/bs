from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import cv2
import matplotlib.pyplot as plt
import itertools
import numpy as np
import time
from define import *


# 用于没有csv的情况
# def get_input_xy(src=[]):
#     pre_x = []
#     true_y = []
#     class_indices={"miao":0,
#                    "yi":1,
#                    "gejia":2,
#                    "menggu":3,
#                    "chaoxian":4,
#                   }
#     for s in src:
#         input = cv2.imread(s)
#         input = cv2.resize(input, (224, 224))
#         pre_x.append(input/127.5-1)

#         fn = os.path.split(s)[0][10:]
#         y = class_indices.get(fn)
#         true_y.append(y)
#     pre_x = np.array(pre_x)
#     return pre_x, true_y


# 绘制混淆矩阵
def plot_sonfusion_matrix(cm, Name, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(6, 5))
    plt.rcParams['savefig.dpi'] = 100  # 图片像素
    plt.imshow(cm, interpolation='nearest', vmax=1, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=35)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j] > 0:
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')
    plt.gcf().subplots_adjust(bottom=0.18)
    plt.savefig('hxjz_' + Name + '.jpg')


def myCMPlot(model=None, experiment=None):
    CMdf = pd.read_csv("dataset.csv")
    CMdf = CMdf[CMdf.clothes == 'minzu']
    val_pics = CMdf[CMdf.type == 'test']

    if model is None:
        wtname = get_str("载入模型名称", "Dense")
        wtpath = "log/model_save/" + wtname + ".h5"
        print("载入" + wtpath)
        model = load_model(wtpath)
    else:
        wtname = experiment
    label = {
        0: "苗",
        1: "彝",
        2: "革家",
        3: "蒙古",
        4: "朝鲜",
    }
    # 网络预测
    rl = []
    pred_y = []
    indexls = []
    t0 = time.clock()
    for pic in val_pics.pic_path:
        indexls.append(val_pics.index[(val_pics.pic_path == pic)].tolist()[0])
        pic = np.float32(cv2.resize(cv2.imread(pic), (224, 224))) / 127.5 - 1
        r = model.predict(np.array([pic]))[0]
        rl.append(r)
        pred_y.append(np.argmax(r))
    print("运算时间", round(time.clock() - t0, 3))
    true_y = list(val_pics.category_code)
    # 计算混淆矩阵
    confusion_mat = confusion_matrix(true_y, pred_y, labels=[0, 1, 2, 3, 4])
    cmp = confusion_mat / 40  # 概率形式
    print(confusion_mat)
    # 绘图
    plot_sonfusion_matrix(cmp, wtname, ['miao', 'yi', 'gejia', 'menggu', 'chaoxian'])

    # 错误检查
    err = []
    for i in range(len(rl)):
        if int(val_pics.category_code[indexls[i]]) != np.argmax(rl[i]):
            print(i + 1, "预测为",
                  label[np.argmax(rl[i])],
                  "  ",
                  val_pics.name[indexls[i]],
                  "  预测错误")
            err.append(val_pics.name[indexls[i]])
    print("预测错误个数：", len(err), "  正确率", 1 - len(err) / 200)


if __name__ == '__main__':
    myCMPlot()
