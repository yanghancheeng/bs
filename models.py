from tensorflow.keras.layers import Layer, Multiply, Flatten, Dense, GlobalAveragePooling2D  # , Dropout
# from tensorflow.keras.layers import Reshape, Conv2D, GlobalMaxPooling2D, Add, Activation
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.nasnet import NASNetLarge, NASNetMobile
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras import Model


# 特征提取器选取与初始化
def get_Mybackbone(name, *args, **kwargs):
    return eval(name)(*args, **kwargs)


# 构建模型
def build_model(backbone, classes):
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(classes, activation='sigmoid', name='predict')(x)
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

