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

def load_data(d_path):
    return [(x['text_normd'],x['label']) for x in LoadJsons(d_path)]

def label2text(dp):
    train_data = load_data(d_path=dp)
    label2text = defaultdict(list)
    for d in train_data:
        for l in d[1]:
            label2text[llist.get_id(l)].append(d[0])
    return label2text
lab2txt = [label2text(os.path.join(datadir, '%s_normd.json') % tp) for tp in ['train', 'val']]  # 0为训练集的label2text字典，1为验证集

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
    def __init__(self,num_pairs:int,low_freq_total_weight:float,lab2tdic:dict,freq_sep:int):
        super().__init__()
        self.data = []
        rs = np.random.rand(num_pairs)
        
        
        # weights = [low_freq_total_weight/milen]*milen+[(1-low_freq_total_weight)/malen]*malen
        #samples = random.choices(minor_labels,weights=weights,k=2*num_pairs)  
        for i in range(num_pairs):
            if rs[i]> 0.5:
                
                label = random.choices(range(1,llist.get_num()),weights=[(1-low_freq_total_weight)/(loc(freq_sep,llist)-1)]*(loc(freq_sep,llist)-1)+
                    [low_freq_total_weight/(llist.get_num()-loc(freq_sep,llist))]*(llist.get_num()-loc(freq_sep,llist)),k=1)[0]
                [tx1,tx2] = random.choices(lab2tdic[label],weights=[1/len(lab2tdic[label])]*len(lab2tdic[label]),k=2)
                self.data.append([tx1,tx2,torch.tensor([1])])
            else:
                labels = random.choices(range(1,llist.get_num()),weights=[(1-low_freq_total_weight)/(loc(freq_sep,llist)-1)]*(loc(freq_sep,llist)-1)+
                    [low_freq_total_weight/(llist.get_num()-loc(freq_sep,llist))]*(llist.get_num()-loc(freq_sep,llist)),k=2)
                tx1 = random.choices(lab2tdic[labels[0]],weights=[1/len(lab2tdic[labels[0]])]*len(lab2tdic[labels[0]]),k=1)[0]
                tx2 = random.choices(lab2tdic[labels[1]],weights=[1/len(lab2tdic[labels[1]])]*len(lab2tdic[labels[1]]),k=1)[0]
                if labels[0] == labels[1]:
                    y = torch.tensor([1])
                else: y = torch.tensor([0])
                self.data.append([tx1,tx2,y])
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
# pairs = StcPairSet(100,0.5,lab2txt[0],20) # test sample
# print(pairs[0])
