# 需要生成成对数据
import os,random,torch
from torch.utils.data import Dataset,DataLoader
from ljqpy import LoadJsons
from sortlabel import TokenList,loc
# 生成样本列表
savefile='sortlabel.txt'
sourcefile='train.json'
f = lambda x:x['label']
llist = TokenList(file=savefile,source=LoadJsons(sourcefile),func=f,low_freq=1,save_low_freq=1)  # 根据12.5的任务4修改为只包含高频标签，如>20

datadir = './'
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
x_min, x_mar = sep_data_byfreq(os.path.join(datadir, 'train_normd.json'),20)  # 只加载了训练集
# milen, malen = len(x_min),len(x_mar)
# print(milen,malen)
class StcPairSet(Dataset):
    def __init__(self,minor_labels:list,major_labels:list,num_pairs:int,low_freq_total_weight:float):
        super().__init__()
        self.data = []
        milen, malen = len(minor_labels),len(major_labels)
        minor_labels.extend(major_labels)
        weights = [low_freq_total_weight/milen]*milen+[(1-low_freq_total_weight)/malen]*malen
        samples = random.choices(minor_labels,weights=weights,k=2*num_pairs)  
        for i in range(num_pairs):
            found = False
            for label in samples[2*i][1]:
                if label in samples[2*i+1][1] and found == False:
                    self.data.append([samples[2*i][0],samples[2*i+1][0],torch.tensor([1])])
                    found == True
            if found == False: self.data.append([samples[2*i][0],samples[2*i+1][0],torch.tensor([0])])
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
# pairs = StcPairSet(x_min,x_mar,5,0.5) # test sample
# print(pairs[0])
