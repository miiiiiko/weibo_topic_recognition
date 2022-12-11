import sys
sys.path.append('.')
sys.path.append('./base4')
from base4.main import load_data,DataLoader,llist,label2vec, cal_hour,tokenizer
from sklearn import metrics
from base4.model import Model4
from base1.model import Model1
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import time
from tqdm import tqdm
import numpy as np
from base4.sortlabel import loc
import math

# label2text = defaultdict(list) 
# label2text[0] = []
# train_data = load_data('./dataset/train_normd.json')
# train_ds = MyDataset(train_data,requires_index=True)
# choose text from train_ds

def get_lab2text(d_path,k):
    label2text = defaultdict(list) 
    label2text[0] = []
    train_data = load_data(d_path)
    for d in train_data:
        for l in d[1]:
            label2text[llist.get_id(l)].append(d[0])
    # select k text for each label
    for i,text_l in label2text.items():
        if text_l:
            label2text[i] = random.choices(text_l,k=k)
    return label2text


# 该collate函数用来加载训练集上挑出来的句子
def collate_c_fn(batch):
    # text,label = [],[]
    # for d in batch:
    #     text.append(d[0])
    #     label.append(d[1])
    z = tokenizer([d[0] for d in batch],return_tensors='pt',truncation=True, max_length=256,padding=True)
    return (z.input_ids,
            z.attention_mask,
            z.token_type_ids,
            torch.cat([d[1] for d in batch], 0)
    )


# 加载验证集
def collate_val_fn(batch):
    # text,label = [],[]
    # for d in batch:
    #     text.append(d[0])
    #     label.append(d[1])
    z = tokenizer([d[0] for d in batch],return_tensors='pt',truncation=True, max_length=256,padding=True)
    return (z.input_ids,
            z.attention_mask,
            z.token_type_ids,
            torch.stack([d[1] for d in batch], 0)
    )



class Val_DS(Dataset):
    def __init__(self,d_path):
        super().__init__()
        data = load_data(d_path)
        self.data = []
        for d in data:
            label = label2vec(d[1],dims= llist.get_num())
            label = torch.from_numpy(label)
            self.data.append([d[0],label])


    def __getitem__(self, index):
        # [text,label_id]
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


class Text_Contrast_DS(Dataset):
    def __init__(self,d_path,k=1,n=0):
        super().__init__()
        label2text = get_lab2text(d_path,k)
        self.data = []
        # print(len(label2text))
        for i in range(n,len(label2text)):
            text_l = label2text[i]
            for t in text_l:
                self.data.append([t,torch.tensor([i])])


    def __getitem__(self, index):
        # [text,label_id]
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

# ds = Text_DS('你好',label2text)
# print(len(ds))
def get_embedding(model,d_path,k=3,n=0,batch_size=64,device = torch.device('cuda')):
    # d_path = './dataset/train_normd.json'
    ds = Text_Contrast_DS(d_path,k,n=n)
    dl = DataLoader(ds,collate_fn=collate_c_fn,batch_size=batch_size)
    rep = []
    indices = []
    # res = []
    model = model.to(device)
    # out_path = './output/base4/label_respresentation.pt'
    with torch.no_grad():
        for ditem in dl:
            input_ids,attention_mask,token_type_ids, zz = ditem[0].to(device), ditem[1].to(device),ditem[2].to(device),ditem[3]
            # xx,zz = d
            # xx = xx.to(device)
            yy = model(input_ids,attention_mask,token_type_ids)
            rep.append(yy)
            indices.append(zz)
    rep = torch.cat(rep)
    rep = F.normalize(rep,dim=1).cpu()
    indices = torch.cat(indices)
    # print(indices)
    return rep,indices


# model = Model()
# model.load_state_dict(torch.load('./output/base4/cls_base4.pt'))
# rep,label = get_embedding(model,'./dataset/train_normd.json')

def get_weights(n):
    return [0]*loc(5,llist) + [math.sin(0.5*i/(1400-loc(5,llist))*math.pi) for i in range(1400-loc(5,llist))]
    # return 1350*[0] + [1 for i in range(1,51)]
    #　return [i/1400 for i in range(1,1401)]
    # return [0]*1300 + [i/100 for i in range(1,101)]
    # return [1]*llist.get_num()
    # return [0]*(loc(5,llist))+[math.cos(i/(10*(1400-loc(5,llist)))) for i in range(1400-loc(5,llist),0,-1)]
    # return [0]*llist.get_num()



class Predictor:
    def __init__(self, model1,model2,mfile1,mfile2,d_path,k=3,n=0,batch_size = 128, get_weights = get_weights,device = torch.device('cuda')):
        self.n = n
        self.k = k
        self.model1 = model1.to(device)
        self.model2 = model2.to(device)
        self.device = device
        self.model1.load_state_dict(state_dict = torch.load(mfile1, map_location=device))
        self.model2.load_state_dict(state_dict = torch.load(mfile2, map_location=device))
        self.model1.eval()
        self.model2.eval()
        self.rep, self.indices = get_embedding(self.model1, d_path, k, n, batch_size, device)
        self.weights = get_weights(llist.get_num())
        

    # # 测试集上对句子列表预测标签
    # def predict(self, sents, threshold=0.5):
    #     scores = torch.zeros(len(sents),llist.get_num())
    #     xx = tokenizer(sents, return_tensors='pt',truncation=True, max_length=256,padding=True)
    #     input_ids = xx.input_ids.to(self.device)
    #     attention_mask = xx.attention_mask.to(self.device)
    #     token_type_ids = xx.token_type_ids.to(self.device)

    #     model = self.model.to(self.device)
    #     with torch.no_grad():
    #         yy = F.normalize(model(input_ids,attention_mask,token_type_ids)).cpu()
    #         cosine_scores = torch.mm(yy,self.rep.T)
    #         # print(cosine_scores.shape)
    #         for i in range(len(cosine_scores[0])):
    #             scores[:,self.indices[i]] += cosine_scores[:,i]
    #     scores = (scores/self.k > threshold).float()
    #     # print(scores)
    #     ret = []
    #     for z in scores:
    #         indices = z.nonzero().squeeze(1)
    #         ret.append([llist.get_token(i) for i in indices])
    #     return ret,scores
       
        
    # 开发集上计算f1
    def eval_on_val(self,val_dl,threshold):
        threshold = torch.tensor([threshold])
        pbar = tqdm(val_dl)
        self.model1.eval()
        self.model2.eval()
        model1 = self.model1.to(self.device)
        model2 = self.model2.to(self.device)
        yt,yp = [],[] 
        # scores = []
        with torch.no_grad():
            for ditem in pbar:
                scores = torch.zeros(len(ditem[0]),llist.get_num())
                input_ids,attention_mask,token_type_ids, zz = ditem[0].to(self.device), ditem[1].to(self.device),ditem[2].to(self.device),ditem[3]
                yy1 = F.normalize(model1(input_ids,attention_mask,token_type_ids)).cpu()
                cosine_scores = torch.mm(yy1,self.rep.T)
                yy2 = model2(input_ids).cpu()
                for i in range(len(cosine_scores[0])):
                    scores[:,self.indices[i]] += self.weights[self.indices[i]]*cosine_scores[:,i]
                scores = scores/self.k
                # print(scores)
                for i in range(llist.get_num()):
                    scores[:,i] += (1-self.weights[i])*yy2[:,i]
                scores = (scores > threshold).float() 

                yt.append(zz)
                yp.append(scores)
                # print(yp)
            # print(len(yt),len(yp))
            # print(yt[0].shape)
            # print(yp[0].shape)
            yt = torch.cat(yt)
            yp = torch.cat(yp)
            # print(yt.shape,yt[:,loc(self.f,llist):].shape)
            # print(type(yt))
            # print(type(yp))
            # print(yt.shape,yp.shape,yt.dtype,yp.dtype)
            # yt,yp = torch.int64(yt),torch.int64(yp)
            pbar.close()
            f1 = metrics.f1_score(yt,yp,average='samples')
            # prec = metrics.precision_score(yt,yp,average='samples')
            # micro_low_freq_f1 = metrics.f1_score(yt,yp,average='samples')
            prec = metrics.precision_score(yt,yp,average='samples')
            reca = metrics.recall_score(yt,yp,average='samples')
            print(f"example_f1: {f1:.4f}, prec: {prec:.4f}, rec: {reca:.4f}")
        return 


model1 = Model4()
model2 = Model1()
mfile1 = './output/base4/cls_base4.pt'
mfile2 = './output/base1/256base1.pt'
d_path = './dataset/train_normd.json'
n = loc(10,llist)
Pred = Predictor(model1,model2,mfile1,mfile2,d_path,n=n)
# print(Cosine_sim.predict([ "左航ZH超话 养成系追星不是投资也不是赌博,是我在陪左航长大 @TF家族-左航 ","我还挺喜欢黄太太的 不是大圣人会为自己儿女有私心，但是在大是大非上从不出错还很聪明"]))
val_ds = Val_DS('./dataset/val_normd.json')
val_dl = DataLoader(val_ds,collate_fn=collate_val_fn,batch_size=128)
threshold = [0.5]* n + [0.8]*(1400-n)
# threshold = [0.47]*1400
Pred.eval_on_val(val_dl, threshold=threshold)
# for d in val_dl:
#     print(d[0].shape)
#     print(d[-1].shape)
#     break