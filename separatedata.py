import json,random
import unicodedata
from ljqpy import LoadJsons,SaveJsons

def sep_data(file_path:str):
    data = []
    for xx in LoadJsons(file_path):
        xx['text_normd'] = xx['text'].replace('\u200b','')
        xx['text_normd'] = unicodedata.normalize('NFKC', xx['text_normd'])
        data.append(xx)
        
    random.shuffle(data)
    train = data[5000:]; val = data[:5000]
    SaveJsons(train,'train_normd.json')
    SaveJsons(val,'val_normd.json')
if __name__ == '__main__':
    random.seed(1305)
    sep_data('train.json')