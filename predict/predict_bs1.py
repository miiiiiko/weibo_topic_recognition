import sys
sys.path.append('.')
sys.path.append('./base4')
import torch
from model import Model1
from base4 import model
from main import tokenizer,llist,MyDataset,collate_fn,load_data
import numpy as np
from torch.utils.data import DataLoader
from sortlabel import sep_point,label2id,label2vec

import json
from sklearn.metrics import *
from merge_func import Val0_Classify,Rules
from tqdm import tqdm
from base4.predict import get_embedding

class Multi_label:
    def __init__(self, model,model4,mfile,llist,device):
        self.model = model.to(device)
        self.device = device
        self.model.load_state_dict(state_dict = torch.load(mfile, map_location=device))
        self.model.eval()
        self.model4 = model4
        self.model4.load_state_dict(state_dict = torch.load('output/base4/cls_base4.pt', map_location=device))
        self.tl = llist
        self.k = 3
        self.f = 15000
        
        batch_size = 128
        self.rep = get_embedding(self.model4, 'dataset/train_normd.json', self.k, self.f, batch_size, self.device)
        self.model4.to(device)
        

    # 测试集直接输出标签list
    def predict(self, sents,true_label):
        s = tokenizer(sents, return_tensors='pt', padding=True) # [text1,text2,...]
        xs = s.input_ids
        attm = s.attention_mask
        tti = s.token_type_ids
        xs = xs.to(self.device)
        predsc_l = []
        truesc_l = []
        label_id = label2id(true_label,llist)
        yy = torch.from_numpy(label2vec(true_label,llist))
        with torch.no_grad():
            scores = self.model(xs).detach().cpu()
            
            zz = (scores > 0.5).float().cpu()
        
        
        for idx,z in enumerate(zz):
            predscore = None
            if sum(z) == torch.tensor(0):
            # if Rules(z,scores[idx], self.tl).rule1():
                z, predscore= Val0_Classify(xs[idx],attm[idx],tti[idx],self.rep,self.k,self.f,self.model,self.model4,mode='argmax').fun()
            if any(yy[idx] != z): 
                if predscore:
                    predsc_l.append(predscore.item())
                else: 
                    pred_id = z.nonzero().squeeze(1)
                    predsc_l.append([scores[idx,i].item() for i in pred_id])
                truesc_l.append([scores[idx,i].item() for i in label_id])
            indices = z.nonzero().squeeze(1)
            ret = [self.tl.get_token(i) for i in indices]
        return ret , truesc_l, predsc_l



    def eval_on_val(self,val_loader,x_l=None):
		# val_loader: requires_index = True
        yt, yp, bad_case_indices = [], [], []
        self.model.eval()
        pbar = tqdm(val_loader, total=len(val_loader))
        with torch.no_grad():
            for xx,attm,tti, yy,indices in pbar:
                xx, yy = xx.to(self.device), yy
                # print(xx.shape)
                scores = self.model(xx).detach().cpu()
                zz = (scores > 0.5).float().cpu()
                # i_l = torch.tensor([any(yy[i] != zz[i])*1 for i in range(len(yy))]).nonzero().squeeze(1)
                
                # bad_case_indices.extend([indices[i] for i in i_l ])
                
                for y in yy: yt.append(y)
                for idx,z in enumerate(zz): 
                    if sum(z) == torch.tensor(0):
                    # if Rules(z,scores[idx], self.tl).rule1():
                        z, predscore = Val0_Classify(xx[idx],attm[idx],tti[idx],self.rep,self.k,self.f,self.model,self.model4,mode='argmax').fun()
                    if any(yy[idx] != z): 
                        
                        bad_case_indices.append(indices[idx])
                        
                    yp.append(z)
                
            yt = torch.stack(yt,0)
            yp = torch.stack(yp,0)
            accu = accuracy_score(yt,yp)
            prec = precision_score(yt,yp,average='samples',zero_division=0)
            reca = recall_score(yt,yp,average='samples',zero_division=0)
            f1 = f1_score(yt,yp,average='samples',zero_division=0)
            print(f'Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.5f}')
            res = []
            if x_l is not None:
                for x in x_l:
                    y1_t = yt[:,:x]
                    y1_p = yp[:,:x]
                    y2_t = yt[:,x:]
                    y2_p = yp[:,x:]
                    res.append([f1_score(y1_t,y1_p,average='samples',zero_division = 0),f1_score(y2_t,y2_p,average='samples',zero_division = 0)])
        return bad_case_indices, res


# model.load_state_dict(torch.load(mfile), map_location=device)
mfile = 'output/base1/256base1.pt'
model1 = Model1()
model4 = model.Model4()
ml = Multi_label(model1,model4,mfile,llist,torch.device('cuda' if torch.cuda.is_available() else "cpu"))
# x_l = sep_point([1,2,5,100]) 
# print(x_l)
# print(ml.predict([" 我还挺喜欢黄太太的 不是大圣人会为自己儿女有私心,但是在大是大非上从不出错还很聪明 "]))
# predictor = ml.predict
val_data = load_data('dataset/val_normd.json')
val_ds = MyDataset(val_data,requires_index=True)
val_dl = DataLoader(val_ds,batch_size=32,collate_fn=collate_fn)
bad_case_indices, res = ml.eval_on_val(val_dl)
print(res)

fw = open('bad_case_argmax_only.json', 'w', encoding='utf-8')
for i in bad_case_indices:
    l = {}
    l['text'] = val_data[i][0]
    l['true_label'] = val_data[i][1]
    l['pred_label'] ,l['true_score'], l['pred_score']= ml.predict(l['text'],l['true_label'])
    
    l = json.dumps(l, ensure_ascii=False)
    fw.write(l + '\n')
fw.close()
