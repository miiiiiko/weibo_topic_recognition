import torch
from model import Model,Model_large
from main import llist, collate_fn, load_data
from main import tokenizer as tokenizer1
from main import MyDataset as MyDataset1
from main2 import tokenizer as tokenizer2
from main2 import MyDataset as MyDataset2
import numpy as np
from torch.utils.data import DataLoader
from sortlabel import sep_point
import pandas as pd
import json
from sklearn.metrics import *


class Multi_label:
    def __init__(self, model1, mfile1, model2, mfile2, llist, device):
        self.model1 = model1.to(device)
        self.model2 = model2.to(device)
        self.device = device
        self.model1.load_state_dict(state_dict=torch.load(mfile1, map_location=device))
        self.model1.eval()
        self.model2.load_state_dict(state_dict=torch.load(mfile2, map_location=device))
        self.model2.eval()
        self.tl = llist

    # 测试集输出标签list
    def predict(self, sents):
        xs1 = tokenizer1(sents, return_tensors='pt', padding=True).input_ids  # [text1,text2,...]
        xs1 = xs1.to(self.device)
        xs2 = tokenizer2(sents, return_tensors='pt', padding=True).input_ids
        xs2 = xs2.to(self.device)
        with torch.no_grad():
            zz = ((self.model1(xs1).detach().cpu()+self.model2(xs2).detach().cpu())/2 > 0.5).float().cpu()
        ret = []
        for z in zz:
            indices = z.nonzero().squeeze(1)
            ret.append([self.tl.get_token(i) for i in indices])
        return ret

    def eval_on_val(self, val_loader1, val_loader2, x_l=None):
        # val_loader: requires_index = True
        yt, yp, bad_case_indices = [], [], []
        self.model1.eval()
        self.model2.eval()
        with torch.no_grad():
            for xx, yy, indices in val_loader1:#这里要怎么加入val_loader2呢
                xx, yy = xx.to(self.device), yy
                # print(xx.shape)
                zz = ((self.model1(xx).detach().cpu()+self.model2(xx).detach().cpu())/2 > 0.5).float().cpu()
                i_l = torch.tensor([any(yy[i] != zz[i]) * 1 for i in range(len(yy))]).nonzero().squeeze(1)
                bad_case_indices.extend([indices[i] for i in i_l])
                for y in yy: yt.append(y)
                for z in zz: yp.append(z)
            yt = torch.stack(yt, 0)
            yp = torch.stack(yp, 0)
            res = []
            if x_l is not None:
                for x in x_l:
                    y1_t = yt[:, :x]
                    y1_p = yp[:, :x]
                    y2_t = yt[:, x:]
                    y2_p = yp[:, x:]
                    res.append([f1_score(y1_t, y1_p, average='samples', zero_division=0),
                                f1_score(y2_t, y2_p, average='samples', zero_division=0)])
        return bad_case_indices, res


# model.load_state_dict(torch.load(mfile), map_location=device)
mfile1 = 'C:/Users/liuweican/人民网比赛/300base1cls.ckpt'
model1 = Model()
mfile2 = 'C:/Users/liuweican/人民网比赛/largebase1.ckpt'
model2 = Model_large()

ml = Multi_label(model1, mfile1, model2, mfile2, llist, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
x_l = sep_point([1, 2, 5, 100])
print(x_l)

val_data = load_data('C:/Users/liuweican/人民网比赛/val_normd.json')
val_ds1 = MyDataset1(val_data, requires_index=True)
val_ds2 = MyDataset2(val_data)
val_dl1 = DataLoader(val_ds1, batch_size=32, collate_fn=collate_fn)
val_dl2 = DataLoader(val_ds2, batch_size=32, collate_fn=collate_fn)
bad_case_indices, res = ml.eval_on_val(val_dl1, val_dl2, x_l)
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