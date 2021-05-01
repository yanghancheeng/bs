from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D  # , Dropout
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras import Model
from labs import *

# 特征提取器结构选择字典
backbones = {}
#  {backbone_name: VGG16 for backbone_name in name_vgg_list}  生成如下:
#  {'VGG':VGG16, 'vgg':VGG16, 'VGG16':VGG16, 'vgg16':VGG16} -> 增强代码健壮性
backbones.update({bk_name: VGG16 for bk_name in l_name_vgg})  # VGG16是函数名，是键值对的值，以下类似
backbones.update({bk_name: VGG19 for bk_name in l_name_vgg19})
backbones.update({bk_name: Xception for bk_name in l_name_Xcep})
backbones.update({bk_name: DenseNet121 for bk_name in l_name_DenseNet})
backbones.update({bk_name: DenseNet169 for bk_name in l_name_DenseNet169})
backbones.update({bk_name: DenseNet201 for bk_name in l_name_DenseNet201})
backbones.update({bk_name: MobileNet for bk_name in l_name_MobileNet})


# 特征提取器选取与初始化
def get_Mybackbone(name, *args, **kwargs):
    return backbones[name](*args, **kwargs), name


# 构建模型
def build_model(backboneInfo, classes):
    backbone = backboneInfo[0]
    name = backboneInfo[1]
    x = backbone.output
    if name == "vgg":
        x = Flatten(name='flatten')(x)
    else:
        x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(classes, activation='sigmoid')(x)
    model = Model(inputs=backbone.input, outputs=predictions)
    return model


# 通用分类模型获取器，要求，输入为图片，输出为类别节点，中部黑盒模型
def get_classifyModels(model_name,
                       input_shape=(224, 224, 3),
                       encoder_weights='imagenet',
                       classes=10,
                       ):
    backbone = get_Mybackbone(model_name,
                              input_shape=input_shape,
                              weights=encoder_weights,
                              include_top=False)
    model = build_model(backbone,
                        classes, )
    return model


# 复杂模型获取器，模型可能有多个输入，多个输出
def get_complexModels():
    pass  # 未实现
