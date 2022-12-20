import json,random
import unicodedata
from ljqpy import LoadJsons,SaveJsons

def sep_data(file_path:str):  # 将数据随机打乱并切分成训练集和验证集
    data = []
    for xx in LoadJsons(file_path):  # 数据格式.json
        xx['text_normd'] = xx['text'].replace('\u200b','')
        xx['text_normd'] = unicodedata.normalize('NFKC', xx['text_normd'])  # 同时清洗数据，并保存到新的字段中
        data.append(xx)
        
    random.shuffle(data)
    train = data[5000:]; val = data[:5000]
    SaveJsons(train,'train_normd.json')
    SaveJsons(val,'val_normd.json')
if __name__ == '__main__':
    random.seed(1305)
    sep_data('train.json')
