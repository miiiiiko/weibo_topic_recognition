import random,torch
from torch.utils.data import Dataset
from ljqpy import LoadJsons
from sortlabel import TokenList,loc
from collections import defaultdict
import numpy as np
# 生成样本列表
savefile='sortlabel.txt'
sourcefile='./dataset/train.json'
f = lambda x:x['label']
llist = TokenList(file=savefile,source=LoadJsons(sourcefile),func=f,low_freq=1,save_low_freq=1)  # 根据12.5的任务4修改为只包含高频标签，如>20)
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

# lab2data = label2data('./dataset/train_normd.json')


def sentence_pair_gen(lab2data,weights):
    while True:
        rs = np.random.uniform()
        if rs> 0.5:
            label = random.choices(range(1,llist.get_num()),weights=weights,k=1)[0]
            sample1,sample2 = random.choices(lab2data[label],k=2)
        else:
            labels = random.choices(range(1,llist.get_num()),weights=weights,k=2)
            sample1 = random.choices(lab2data[labels[0]],k=1)[0]
            sample2 = random.choices(lab2data[labels[1]],k=1)[0] 
        if sample1[0] != sample2[0]: 
            y = torch.tensor([1]) if set(sample1[1])&set(sample2[1]) else torch.tensor([0])
            yield [sample1[0],sample2[0],y]
                
# x_min, x_mar = sep_data_byfreq(os.path.join(datadir, 'train_normd.json'),20)  # 只加载了训练集
# milen, malen = len(x_min),len(x_mar)
# print(milen,malen)
class StcPairSet(Dataset):
    def __init__(self,gen,num_pairs:int,low_freq_total_weight:float,dpath:str,freq_sep:int):
        super().__init__()
        self.gen = gen
        self.num = num_pairs
        self.lab2data = label2data(dpath)
        high_freq_num = loc(freq_sep,llist)-1
        low_freq_num = llist.get_num()-loc(freq_sep,llist)
        high_freq_weight = (1-low_freq_total_weight)/high_freq_num
        low_freq_weight = low_freq_total_weight/low_freq_num
        self.weights = [high_freq_weight]*high_freq_num + [low_freq_weight]*low_freq_num

    def __getitem__(self, index):
        return next(self.gen(self.lab2data,self.weights))

    def __len__(self):
        return self.num


pairs_vs = StcPairSet(sentence_pair_gen,1000,0.3,'/home/qsm22/weibo_topic/dataset/train_normd.json',20)
# print(len(pairs_ts))
# pairs_vs = StcPairSet(10000,0.3,'/home/qsm22/weibo_topic_recognition/dataset/train_normd.json',20)# test sample
# print(pairs_ts[0])
# print(pairs[0])
# cnt = 0
# for i in range(len(pairs_ts)):
#     d = pairs_ts[i]
#     cnt += d[2]
# print(cnt)

