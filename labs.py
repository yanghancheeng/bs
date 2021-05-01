import pandas as pd
from define import dictsAdd

# 共用参数选择
com_Spl = {'TVSpl': 0.9, 'sed': 0}
com_dec = {'decay': 0.001}
com_no_Aug = {'inPre': 1, 'Aug': 0}
com_is_Aug = {'inPre': 0, 'Aug': 4}

# 注册实验
labList = [
    # 实验0：VGG模型，随机权重，无任何数据增强
    {'exp': 'VGG_N_N',
     'model': 'VGG', 'wight': None, 'lr': 5e-3,
     'genTyp': 'NTa_NMa', 'inPre': 1, 'Aug': 0},

    # 实验1：VGG模型，随机权重，无任何数据增强
    {'exp': 'VGG_Imag_N',
     'model': 'VGG', 'wight': 'imagenet', 'lr': 1e-5,
     'genTyp': 'NTa_NMa', 'inPre': 1, 'Aug': 0},

    # 实验2：VGG模型，ImageNet权重，传统数据增强
    {'exp': 'VGG_Imag_T',
     'model': 'VGG', 'wight': 'imagenet', 'lr': 1e-5,
     'genTyp': "Ta_NMa", 'inPre': 0, 'Aug': 4},

    # 实验3：VGG模型，ImageNet权重，掩膜数据增强
    {'exp': 'VGG_Imag_M',
     'model': 'VGG', 'wight': 'imagenet', 'lr': 1e-5,
     'genTyp': "NTa_Ma", 'inPre': 0, 'Aug': 4},

    # 实验4：VGG模型，ImageNet权重，混合数据增强
    {'exp': 'VGG_Imag_TM',
     'model': 'VGG', 'wight': 'imagenet', 'lr': 1e-5,
     'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4},

    # 实验5：Xception模型，ImageNet权重，混合数据增强
    {'exp': 'Xcep_Image_TM',
     'model': 'Xception', 'wight': 'imagenet', 'lr': 1e-5,
     'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4},

    # 实验7：DneseNet模型，随机权重，无数据增强
    {'exp': 'Dense_N_N',
     'model': 'DenseNet', 'wight': None, 'lr': 2e-4,
     'genTyp': "NTa_NMa", 'inPre': 1, 'Aug': 0},

    # 实验8：DneseNet模型，ImageNet权重，无数据增强
    {'exp': 'Dense_Image_N',
     'model': 'DenseNet', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "NTa_NMa", 'inPre': 1, 'Aug': 0},

    # 实验9：DneseNet模型，ImageNet权重，传统数据增强
    {'exp': 'Dense_Image_T',
     'model': 'DenseNet', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "Ta_NMa", 'inPre': 0, 'Aug': 4},

    # 实验9：DneseNet模型，ImageNet权重，背景数据增强
    {'exp': 'Dense_Image_M',
     'model': 'DenseNet', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "NTa_Ma", 'inPre': 0, 'Aug': 4},

    # 实验10：DneseNet模型，ImageNet权重，混合数据增强
    {'exp': 'Dense_Imag_TM',
     'model': 'DenseNet', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4},

    # 实验11：DneseNet169模型，ImageNet权重，混合数据增强
    {'exp': 'Dense2_Imag_TM',
     'model': 'DenseNet2', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4},

    # 实验12：DneseNet201模型，ImageNet权重，混合数据增强
    {'exp': 'Dense3_Imag_TM',
     'model': 'DenseNet3', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4},

    # 实验13：MobileNet模型，ImageNet权重，混合数据增强
    {'exp': 'Mobile_Imag_TM',
     'model': 'MobileNet', 'wight': 'imagenet', 'lr': 2e-4,
     'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4},
]

for i in range(len(labList)):
    labList[i] = dictsAdd(labList[i], com_Spl, com_dec)

pdLabLis = pd.DataFrame(labList)
