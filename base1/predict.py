import torch
from model import Model
from main import tokenizer,llist,MyDataset,collate_fn,load_data
import numpy as np
from torch.utils.data import DataLoader
from sortlabel import sep_point
import pandas as pd
import json
from sklearn.metrics import *

class Multi_label:
    def __init__(self, model,mfile,llist,device):
        self.model = model.to(device)
        self.device = device
        self.model.load_state_dict(state_dict = torch.load(mfile, map_location=device))
        self.model.eval()
        self.tl = llist
        

    # 测试集直接输出标签list
    def predict(self, sents):
        xs = tokenizer(sents, return_tensors='pt', padding=True).input_ids # [text1,text2,...]
        xs = xs.to(self.device)
        with torch.no_grad():
            zz = (self.model(xs).detach().cpu() > 0.5).float().cpu()
        ret = []
        for z in zz:
            indices = z.nonzero().squeeze(1)
            ret.append([self.tl.get_token(i) for i in indices])
        return ret



    def eval_on_val(self,val_loader,x_l=None):
		# val_loader: requires_index = True
        yt, yp, bad_case_indices = [], [], []
        self.model.eval()
        with torch.no_grad():
            for xx, yy,indices in val_loader:
                xx, yy = xx.to(self.device), yy
                # print(xx.shape)
                zz = (self.model(xx).detach().cpu() > 0.5).float().cpu()
                i_l = torch.tensor([any(yy[i] != zz[i])*1 for i in range(len(yy))]).nonzero().squeeze(1)
                bad_case_indices.extend([indices[i] for i in i_l ])
                for y in yy: yt.append(y)
                for z in zz: yp.append(z)
            yt = torch.stack(yt,0)
            yp = torch.stack(yp,0)
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
mfile = '/home/qsm22/weibo_topic_recognition/256base1.pt'
model = Model()
ml = Multi_label(model,mfile,llist,torch.device("cuda" if torch.cuda.is_available() else "cpu"))
x_l = sep_point([1,2,5,100]) 
print(x_l)
# print(ml.predict([" 我还挺喜欢黄太太的 不是大圣人会为自己儿女有私心,但是在大是大非上从不出错还很聪明 "]))
# predictor = ml.predict
val_data = load_data('/home/qsm22/weibo_topic_recognition/dataset/val_normd.json')
val_ds = MyDataset(val_data,requires_index=True)
val_dl = DataLoader(val_ds,batch_size=32,collate_fn=collate_fn)
bad_case_indices, res = ml.eval_on_val(val_dl,x_l)
print(res)

fw = open('bad_case.json', 'w', encoding='utf-8')
for i in bad_case_indices:
    l = {}
    l['text'] = val_data[i][0]
    l['true_label'] = val_data[i][1]
    l['pred_label'] = ml.predict(l['text'])[0]
    l = json.dumps(l, ensure_ascii=False)
    fw.write(l + '\n')
fw.close()


# bad_case_indices = ml.eval_on_val(val_dl)
# print(len(bad_case_indices))
# text = val_data[0][0]
# print(text)
# print(ml.predict(text))
# print(val_data[0][1])



