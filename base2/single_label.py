import torch
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os
# from easonsi import utils
# from ljqpy import LoadJsons
# from ljqpy import TokenList
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model_single import Model,tokenizer
import torch.nn as nn
from torch.optim import AdamW
import transformers
from tqdm import tqdm

from sklearn import metrics
import time
import ljqpy
import pt_utils
from ljqpy import LoadJsons
import sortlabel


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
source = ljqpy.LoadJsons('./dataset/train.json')
# tl = sortlabel.TokenList('labellist.txt', source=source, func=lambda x:x['label'], low_freq=1, save_low_freq=1)

# def label2vec(targrtlabels:list,dims= 1400,savefile='sortlabel.txt',sourcefile='train.json',f = lambda x:x['label']):
#     lab_vec = np.zeros(dims)
#     for label in targrtlabels:
#         loc = llist.get_id(label)
#         lab_vec[loc] = 1.0
#     return lab_vec


def load_data(fn):
    # print('loading data')
    return [(x["text_normd"], x["label"]) for x in ljqpy.LoadJsons(fn)]

datadir = './dataset'
# train_dir = '/home/qsm22/weibo_topic_recognition/dataset/train_normd.json'
xys = [load_data(os.path.join(datadir, '%s_normd.json') % tp) for tp in ['train', 'val']]
# tl = ljqpy.TokenList('labellist.txt', source=xys[0], func=lambda x:[x[1]], special_marks=None)
#print(xys[0][0])
# llist = ljqpy.TokenList(file='labellist.txt',source=LoadJsons('./dataset/train.json'),func=lambda x:x['label'])


# def label2vec(targrtlabels:list,dims= 1400):
#     lab_vec = np.zeros(dims)
    
#     for label in targrtlabels:
#         loc = llist.get_id(label)
#         lab_vec[loc] = 1.0
#     return lab_vec

class MyDataset(Dataset):
    def __init__(self,data,maxlen=128):
        # global max_label
        super().__init__()
        self.data = []
        # cnt = 0   
        # prev_time = time.time()     
        for d in data:
            text = tokenizer([d[0]],return_tensors='pt', truncation=True, max_length=maxlen)['input_ids'][0]
            label = sortlabel.label2vec(d[1],dims= 1400)
            label = torch.from_numpy(label)
            fake_label = sortlabel.label2id(d[1])
            for i in range(len(d[1])):
                self.data.append([text,label,torch.tensor([fake_label[i]])])
            # cnt += 1
            # if cnt%10 == 0:
            #     print(time.time()-prev_time)
            #     prev_time = time.time()

    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


# x = ['??????????????????','???????????????','??????']
# y = [[0,1,0],[1,0,1],[1,1,0]]
# data = zip(x,y)
# dataset = MyDataset(data)

# print(max_label)

model_name = 'hfl/chinese-roberta-wwm-ext'



def collate_fn(batch):
    return (nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True),
			torch.stack([x[1] for x in batch], 0), torch.cat([x[2] for x in batch],0)) 



# ds_train, ds_test = MyDataset(xys[0]), MyDataset(xys[1])
# dl_train = torch.utils.data.DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=collate_fn)
# dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, collate_fn=collate_fn)
# for batch in dl_train:
#     print(len(batch))
     

def plot_learning_curve(record):
    # x1 = np.arange(config['total_steps'])
    # x2 = x1[::config['valid_steps']]
    y1 = record['train_loss']
    y2 = record['val_f1']
    y2 = [0 if np.isnan(item) else item for item in y2]
    x1 = np.arange(1,len(y1)+1)
    x2 = x1[1:len(y1)+1:int(len(y1)/len(y2))]
    fig = figure(figsize = (6,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(x1,y1, c = 'tab:red', label = 'train_loss')
    ax2 = ax1.twinx()
    ax2.plot(x2,y2, c='tab:cyan', label='val_f1')
    ax1.set_xlabel('steps')
    ax1.set_ylabel('train_loss')
    ax2.set_ylabel('val_f1')
    plt.title('Learning curve')
    ax1.legend(loc=1)
    ax2.legend(loc=2)
    # plt.show()
    plt.savefig('learning_curve')
    return

def cal_hour(seconds):
    # seconds =35400
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    # print("%d:%02d:%02d" % (h, m, s))
    return "%d:%02d:%02d" % (h, m, s)
    
if __name__ == '__main__':
    record = {"train_loss":[],"val_f1":[]}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    
    ds_train, ds_test = MyDataset(xys[0]), MyDataset(xys[1])
    print('loading data completed')
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=128, shuffle=True, collate_fn=collate_fn)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=128, collate_fn=collate_fn)
    print("dataloader completed")
    model = Model(model_name, 1400).to(device)
    print("finish loading model")
    mfile = 'base2.pt'

    epochs = 1
    total_steps = len(dl_train) * epochs

    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, 1e-4, total_steps)
    loss_func = nn.CrossEntropyLoss()
    start_time = time.time()
    val_time = 0
    def train_func(model, ditem):
        # global train_time
        # t1 = time.time()
        # xx???????????????yy?????????????????????fake??????????????????index
        xx, fake = ditem[0].to(device), ditem[2].to(device)
        zz = model(xx)
        # print(zz.shape,yy.shape)
        loss = loss_func(zz, fake)
        # pred = (zz > 0.5).float()
        # prec = (pred + yy > 1.5).sum()/max(1,pred.sum().item())
        # reca = (pred + yy > 1.5).sum()/max(1,yy.sum().item())
        record["train_loss"].append(loss.item())
        # t2 = time.time()
        # train_time += t2-t1
        # f1 = 2 * prec * reca / (prec + reca)
        return {'loss': loss}

    def test_func(): 
        global val_time
        t1 = time.time()
        yt, yp = [], []
        model.eval()
        with torch.no_grad():
            for d_item in dl_test:
                xx,yy,_ = d_item
                # print(len(dl_test))
                xx, yy = xx.to(device), yy
                # zz = model(xx).detach().cpu().argmax(-1)
                zz_index = model(xx).detach().cpu().argmax(-1).unsqueeze(1)
                zz = torch.zeros_like(yy).scatter(1,zz_index,1).cpu()
                # for y in yy: yt.append(y.item())
                # for z in zz: yp.append(z.item())
                for y in yy: yt.append(y)
                for z in zz: yp.append(z)
            # accu = (np.array(yt) == np.array(yp)).sum() / len(yp)
            # f1 = metrics.f1_score(np.array(yt).astype('int64'), np.array(yp).astype('int64'), average='samples')
            yt = torch.cat(yt,0)
            yp = torch.cat(yp,0)
            accu = (yt == yp).float().mean()
            prec = (yt + yp > 1.5).float().sum() / max(yp.sum().item(),1)
            reca = (yt + yp > 1.5).float().sum() / max(yt.sum().item(),1)
            f1 = 2 * prec * reca / (prec + reca)
            record["val_f1"].append(f1.item())
        print(f'Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.4f}')
        model.train()
        t2 = time.time()
        val_time += t2-t1

    print('Start training!')
    pt_utils.train_model(model, optimizer, dl_train, epochs, train_func, test_func, scheduler=scheduler, save_file=mfile)
    plot_learning_curve(record)
    end_time = time.time()
    total_time = end_time-start_time
    train_time = total_time - val_time
    total_time,train_time,val_time = cal_hour(total_time),cal_hour(train_time),cal_hour(val_time)
    # val_time = cal_hour(val_time)
    # train_time = total_time - val_time
    print(f'Train_time:{train_time}, Val_time:{val_time}, total_time:{total_time}')

    print('done')
    