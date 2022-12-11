from main import load_data,DataLoader,llist,label2vec, cal_hour,tokenizer
from sklearn import metrics
from model import Model
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import time
from tqdm import tqdm
import numpy as np
from sortlabel import loc

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

def collate_c_fn(batch):
    # text,label = [],[]
    # for d in batch:
    #     text.append(d[0])
    #     label.append(d[1])
    z = tokenizer([d for d in batch],return_tensors='pt',truncation=True, max_length=256,padding=True)
    return (z.input_ids,
            z.attention_mask,
            z.token_type_ids,
    )

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
                self.data.append(t)


    def __getitem__(self, index):
        # [text,label_id]
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

# ds = Text_DS('你好',label2text)
# print(len(ds))
def get_embedding(model,d_path,k=3,f=5,batch_size=64,device = torch.device('cuda')):
    # d_path = './dataset/train_normd.json'
    ds = Text_Contrast_DS(d_path,k,n=loc(f,llist))
    dl = DataLoader(ds,collate_fn=collate_c_fn,batch_size=batch_size)
    rep = []
    # indices = []
    # res = []
    model = model.to(device)
    # out_path = './output/base4/label_respresentation.pt'
    with torch.no_grad():
        for ditem in dl:
            input_ids,attention_mask,token_type_ids = ditem[0].to(device), ditem[1].to(device),ditem[2].to(device)
            # xx,zz = d
            # xx = xx.to(device)
            yy = model(input_ids,attention_mask,token_type_ids)
            rep.append(yy)
            # indices.append(zz)
    rep = torch.cat(rep)
    rep = F.normalize(rep,dim=1).cpu()
    # indices = torch.cat(indices)
    # print(indices)
    return rep


class Text_cosine_sim:
    def __init__(self, model,mfile,d_path,k=3,f=5,batch_size = 128, device = torch.device('cuda')):
        self.f = f
        self.k = k
        self.model = model.to(device)
        self.device = device
        self.model.load_state_dict(state_dict = torch.load(mfile, map_location=device))
        self.model.eval()
        self.rep = get_embedding(self.model, d_path, k, f, batch_size, device)
        

    # 测试集上对句子列表预测标签
    def predict(self, sents, threshold=0.5):
        scores = torch.zeros(len(sents),llist.get_num())
        xx = tokenizer(sents, return_tensors='pt',truncation=True, max_length=256,padding=True)
        input_ids = xx.input_ids.to(self.device)
        attention_mask = xx.attention_mask.to(self.device)
        token_type_ids = xx.token_type_ids.to(self.device)

        model = self.model.to(self.device)
        with torch.no_grad():
            yy = F.normalize(model(input_ids,attention_mask,token_type_ids)).cpu()
            cosine_scores = torch.mm(yy,self.rep.T)
            # print(cosine_scores.shape)
            for i in range(len(cosine_scores[0])):
                j = loc(self.f,llist) + i//3
                scores[:,j] += cosine_scores[:,i]
        scores = (scores/self.k > threshold).float()
        # print(scores)
        ret = []
        for z in scores:
            indices = z.nonzero().squeeze(1)
            ret.append([llist.get_token(i) for i in indices])
        return ret,scores
       
        
    # 开发集上计算f1
    def eval_on_val(self,val_dl,threshold=0.7):
        pbar = tqdm(val_dl)
        self.model.eval()
        model = self.model.to(self.device)
        yt,yp = [],[] 
        # scores = []
        with torch.no_grad():
            for ditem in pbar:
                scores = torch.zeros(len(ditem[0]),llist.get_num())
                input_ids,attention_mask,token_type_ids, zz = ditem[0].to(self.device), ditem[1].to(self.device),ditem[2].to(self.device),ditem[3]
                yy = F.normalize(model(input_ids,attention_mask,token_type_ids)).cpu()
                cosine_scores = torch.mm(yy,self.rep.T)
                for i in range(len(cosine_scores[0])):
                    j = loc(self.f,llist) + i//3
                    # print(j)
                    scores[:,j] += cosine_scores[:,i]
                scores = (scores/self.k > threshold).float() 
                yt.append(zz)
                yp.append(scores)
            yt = torch.cat(yt)[:,loc(self.f,llist):]
            yp = torch.cat(yp)[:,loc(self.f,llist):]
            pbar.close()
            f1 = metrics.f1_score(yt,yp,average='samples')
            macro_low_freq_f1 = metrics.f1_score(yt,yp,average='macro')
            micro_low_freq_f1 = metrics.f1_score(yt,yp,average='micro')
            prec = metrics.precision_score(yt,yp,average='micro')
            reca = metrics.recall_score(yt,yp,average='micro')
            print(f"Macro_f1: {macro_low_freq_f1 :.4f},  Micro_f1: { micro_low_freq_f1:.4f}, example_f1: {f1:.4f}, micro_prec: {prec:.4f}, micro_rec: {reca:.4f}")
        return 


model = Model()
mfile = './output/base4/cls_base4.pt'
#model.load_state_dict(state_dict = torch.load(mfile, map_location='cuda'))
d_path = './dataset/train_normd.json'
#get_embedding(model,d_path)
Cosine_sim = Text_cosine_sim(model,mfile,d_path)
# print(Cosine_sim.predict([ "左航ZH超话 养成系追星不是投资也不是赌博,是我在陪左航长大 @TF家族-左航 ","我还挺喜欢黄太太的 不是大圣人会为自己儿女有私心，但是在大是大非上从不出错还很聪明"]))
val_ds = Val_DS('./dataset/val_normd.json')
val_dl = DataLoader(val_ds,collate_fn=collate_val_fn,batch_size=256)
Cosine_sim.eval_on_val(val_dl, threshold=0.8)
# for d in val_dl:
#     print(d[0].shape)
#     print(d[-1].shape)
#     break