import os,random,torch
from torch.utils.data import Dataset,DataLoader
from ljqpy import LoadJsons
from sortlabel import TokenList,loc
from collections import defaultdict
import numpy as np
# 生成样本列表
savefile='sortlabel.txt'
sourcefile='./dataset/train.json'
f = lambda x:x['label']
llist = TokenList(file=savefile,source=LoadJsons(sourcefile),func=f,low_freq=1,save_low_freq=1)  # 根据12.5的任务4修改为只包含高频标签，如>20

datadir = './dataset'

def load_data(d_path):
    return [(x['text_normd'],x['label']) for x in LoadJsons(d_path)]

def label2data(dp):
    train_data = load_data(d_path=dp)
    label2data = defaultdict(list)
    for d in train_data:
        for l in d[1]:
            label2data[llist.get_id(l)].append(d)
    return label2data
# lab2txt = [label2text(os.path.join(datadir, '%s_normd.json') % tp) for tp in ['train', 'val']]  # 0为训练集的label2text字典，1为验证集

def sep_data_byfreq(d_path, freq_sep:int):  # 只针对训练集
    '''
    以频率freq为界，将样本中存在频率小于等于freq的标签的归入一类，否则放入另一类
    '''
    minor_labels = [];major_labels = []
    for x in LoadJsons(d_path):
        label_id = [llist.get_id(token = label) for label in x['label']]   
        if any([id >= loc(freq_sep,llist = llist) for id in label_id]):
            minor_labels.append((x['text_normd'],x['label']))
        else: major_labels.append((x['text_normd'],x['label']))
    return minor_labels, major_labels
# x_min, x_mar = sep_data_byfreq(os.path.join(datadir, 'train_normd.json'),20)  # 只加载了训练集
# milen, malen = len(x_min),len(x_mar)
# print(milen,malen)
class StcPairSet(Dataset):
    def __init__(self,num_pairs:int,low_freq_total_weight:float,dpath:str,freq_sep:int):
        super().__init__()
        self.data = []
        rs = np.random.rand(num_pairs)
        lab2data = label2data(dpath)
        high_freq_num = loc(freq_sep,llist)-1
        low_freq_num = llist.get_num()-loc(freq_sep,llist)
        high_freq_weight = (1-low_freq_total_weight)/high_freq_num
        low_freq_weight = low_freq_total_weight/low_freq_num
        weights = [high_freq_weight]*high_freq_num + [low_freq_weight]*low_freq_num
        # weights = [low_freq_total_weight/milen]*milen+[(1-low_freq_total_weight)/malen]*malen
        #samples = random.choices(minor_labels,weights=weights,k=2*num_pairs)  
        for i in range(num_pairs):
            if rs[i]> 0.5:
                label = random.choices(range(1,llist.get_num()),weights=weights,k=1)[0]
                samples = random.choices(lab2data[label],k=2)
                self.data.append([samples[0][0],samples[1][0],torch.tensor([1])])
            else:
                labels = random.choices(range(1,llist.get_num()),weights=weights,k=2)
                sample1 = random.choices(lab2data[labels[0]],k=1)[0]
                sample2 = random.choices(lab2data[labels[1]],k=1)[0]
                # if labels[0] == labels[1]:
                #     y = torch.tensor([1])
                # else: y = torch.tensor([0])
                y = torch.tensor([1]) if set(sample1[1])&set(sample2[1]) else torch.tensor([0])
                self.data.append([sample1[0],sample2[0],y])
            # found = False
            # for label in samples[2*i][1]:
            #     if label in samples[2*i+1][1] and found == False:
            #         self.data.append([samples[2*i][0],samples[2*i+1][0],torch.tensor([1])])
            #         found == True
            # if found == False: self.data.append([samples[2*i][0],samples[2*i+1][0],torch.tensor([0])])
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
pairs_ts = StcPairSet(100000,0.3,'/home/qsm22/weibo_topic_recognition/dataset/train_normd.json',20)
pairs_vs = StcPairSet(10000,0.3,'/home/qsm22/weibo_topic_recognition/dataset/train_normd.json',20)# test sample
# print(pairs_ts[0])
# print(pairs[0])
# cnt = 0
# for i in range(len(pairs_ts)):
#     d = pairs_ts[i]
#     cnt += d[2]
# print(cnt)

