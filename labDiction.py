import pandas as pd

# def dicUpdate(dic,**dics){
#     for
# }


# 实验0：VGG模型，随机权重，无任何数据增强
d0 = {'exp': 'VGG_None_None',
      'model': 'VGG', 'wight': None, 'lr': 5e-3, 'decay': 0.001,
      'genTyp': 'NTa_NMa', 'inPre': 1, 'Aug': 0, 'TVSpl': 0.9, 'sed': 0}

# 实验1：VGG模型，随机权重，无任何数据增强
d1 = {'exp': 'VGG_Imag_None',
      'model': 'VGG', 'wight': 'imagenet', 'lr': 1e-5, 'decay': 0.001,
      'genTyp': 'NTa_NMa', 'inPre': 1, 'Aug': 0, 'TVSpl': 0.9, 'sed': 0}

# 实验2：VGG模型，ImageNet权重，传统数据增强
d2 = {'exp': 'VGG_Imag_T',
      'model': 'VGG', 'wight': 'imagenet', 'lr': 1e-5, 'decay': 0.001,
      'genTyp': "Ta_NMa", 'inPre': 0, 'Aug': 4, 'TVSpl': 0.9, 'sed': 0}

# 实验3：VGG模型，ImageNet权重，掩膜数据增强
d3 = {'exp': 'VGG_Imag_M',
      'model': 'VGG', 'wight': 'imagenet', 'lr': 1e-5, 'decay': 0.001,
      'genTyp': "NTa_Ma", 'inPre': 0, 'Aug': 4, 'TVSpl': 0.9, 'sed': 0}

# 实验3：VGG模型，ImageNet权重，混合数据增强
d4 = {'exp': 'VGG_Imag_TM',
      'model': 'VGG', 'wight': 'imagenet', 'lr': 1e-5, 'decay': 0.001,
      'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4, 'TVSpl': 0.9, 'sed': 0}

# 实验5：Xception模型，ImageNet权重，混合数据增强
d5 = {'exp': 'Xcep_Image_TM',
      'model': 'VGG', 'wight': 'imagenet', 'lr': 1e-5, 'decay': 0.001,
      'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4, 'TVSpl': 0.9, 'sed': 0}

# 实验6：DneseNet模型，ImageNet权重，混合数据增强
d6 = {'exp': 'Dense_Imag_TM',
      'model': 'Xception', 'wight': 'imagenet', 'lr': 2e-4, 'decay': 0.001,
      'genTyp': "Ta_Ma", 'inPre': 0, 'Aug': 4, 'TVSpl': 0.9, 'sed': 0}
# 实验7：DneseNet模型，随机权重，无数据增强
d7 = {'exp': 'Dense_Non_Non',
      'model': 'DenseNet', 'wight': None, 'lr': 2e-4, 'decay': 0.001,
      'genTyp': "NTa_NMa", 'inPre': 1, 'Aug': 0, 'TVSpl': 0.9, 'sed': 0}

# 实验8：DneseNet模型，随机权重，无数据增强
d8 = {'exp': 'Dense_Image_Non',
      'model': 'DenseNet', 'wight': 'imagenet', 'lr': 2e-4, 'decay': 0.001,
      'genTyp': "NTa_NMa", 'inPre': 1, 'Aug': 0, 'TVSpl': 0.9, 'sed': 0}

labDictionary = {
    0: d0,
    1: d1,
    2: d2,
    3: d3,
    4: d4,
    5: d5,
    6: d6,
    7: d7,
    8: d8,
}

pdLabDic = pd.DataFrame(labDictionary).T
# pd.set_option('display.max_columns', None)
# pd.set_option('max_colwidth', 5)
