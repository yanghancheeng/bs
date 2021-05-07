import pandas as pd


def dictsAdd(dic, *args):
    for it in args:
        dic = dict(list(dic.items()) + list(it.items()))
    return dic


class Labs:
    # 共用参数选择
    com_Spl = {'TVSpl': 0.9, 'sed': 0}
    com_dec = {'decay': 0.001}
    labList = []

    def __init__(self, *args):
        for lab in args:
            self.labList.append(dictsAdd(lab, self.com_Spl, self.com_dec))

    def addLab(self, dic):
        self.labList.append(dictsAdd(dic, self.com_Spl, self.com_dec))

    def addGoodLab(self, *args):
        for lab in args:
            self.labList.append(dictsAdd(lab, self.com_Spl, self.com_dec, {'wight': 'imagenet', 'lr': 2e-4,
                                                                           'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4}))

    def getDataFrame(self):
        return pd.DataFrame(self.labList)

    def updateLr(self, lr):
        if not (lr is None) and lr > 0:
            for i in range(len(self.labList)):
                self.labList[i]["lr"] = lr


# 注册实验
labs = Labs(

    # VGG模型，随机权重，无任何数据增强
    {'exp': 'VGG_N_N',
     'model': 'VGG16', 'wight': None, 'lr': 1e-5,
     'genTyp': 'NTa_NMa', 'inPre': 1, 'Aug': 0},
    # VGG模型，ImageNet权重，无任何数据增强
    {'exp': 'VGG_Imag_N',
     'model': 'VGG16', 'wight': 'imagenet', 'lr': 1e-5,
     'genTyp': 'NTa_NMa', 'inPre': 1, 'Aug': 0},

    # VGG模型，ImageNet权重，传统数据增强
    {'exp': 'VGG_Imag_T',
     'model': 'VGG16', 'wight': 'imagenet', 'lr': 1e-5,
     'genTyp': "Ta_NMa", 'inPre': 0, 'Aug': 4},

    # VGG模型，ImageNet权重，掩膜数据增强
    {'exp': 'VGG_Imag_M',
     'model': 'VGG16', 'wight': 'imagenet', 'lr': 1e-5,
     'genTyp': "NTa_Ma", 'inPre': 0, 'Aug': 4},

    # VGG模型，ImageNet权重，混合数据增强
    {'exp': '`VGG_Imag_TM',
     'model': 'VGG16', 'wight': 'imagenet', 'lr': 1e-5,
     'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4},

    # Xception模型，ImageNet权重，混合数据增强
    {'exp': '`Xcep_Image_TM',
     'model': 'Xception', 'wight': 'imagenet', 'lr': 1e-5,
     'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4},

    # DneseNet模型，随机权重，无数据增强
    {'exp': 'Dense_N_N',
     'model': 'DenseNet121', 'wight': None, 'lr': 2e-4,
     'genTyp': "NTa_NMa", 'inPre': 1, 'Aug': 0},

    # DneseNet模型，ImageNet权重，无数据增强
    {'exp': 'Dense_Image_N',
     'model': 'DenseNet121', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "NTa_NMa", 'inPre': 1, 'Aug': 0},

    # DneseNet模型，ImageNet权重，传统数据增强
    {'exp': 'Dense_Image_T',
     'model': 'DenseNet121', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "Ta_NMa", 'inPre': 0, 'Aug': 4},

    # DneseNet模型，ImageNet权重，背景数据增强
    {'exp': 'Dense_Image_M',
     'model': 'DenseNet121', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "NTa_Ma", 'inPre': 0, 'Aug': 4},

    # DneseNet模型，ImageNet权重，混合数据增强
    {'exp': '`Dense_Imag_TM',
     'model': 'DenseNet121', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4},

    # DneseNet169模型，ImageNet权重，混合数据增强
    {'exp': '`Dense2_Imag_TM',
     'model': 'DenseNet169', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4},

    # DneseNet201模型，ImageNet权重，混合数据增强
    {'exp': '`Dense3_Imag_TM',
     'model': 'DenseNet201', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4},
)

labs.addGoodLab(
    # MobileNet模型，ImageNet权重，混合数据增强
    {'exp': '`Mobile_Imag_TM',
     'model': 'MobileNet'},
    # NASNetLarge，标准参数
    {'exp': '`NASNet_Imag_TM',
     'model': 'NASNetMobile'},
    # NASNetLarge，标准参数
    {'exp': '`ResNet50V2_Imag_TM',
     'model': 'ResNet50V2'},
    # NASNetLarge，标准参数
    {'exp': '`IncepResNetV2_Imag_TM',
     'model': 'InceptionResNetV2'},

)
