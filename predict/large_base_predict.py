import torch
from model import Model1,Model_large
from main import llist, collate_fn, load_data
from main import tokenizer,MyDataset
import numpy as np
from torch.utils.data import DataLoader
from sortlabel import sep_point
# import pandas as pd
import json
from sklearn.metrics import *
from transformers import BertTokenizer
from tqdm import tqdm

tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
tokenizer_l = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')

def collate_fn_base_large(batch):
    z = tokenizer([d[0] for d in batch],return_tensors='pt',truncation=True, max_length=128,padding=True)
    z_l = tokenizer_l([d[0] for d in batch],return_tensors='pt',truncation=True, max_length=128,padding=True)
    return (z.input_ids,
            z_l.input_ids,
            torch.stack([x[1] for x in batch], 0))


class Multi_label:
    def __init__(self, model1, mfile1, model2, mfile2, device):
        self.model1 = model1.to(device)
        self.model2 = model2.to(device)
        self.device = device
        self.model1.load_state_dict(state_dict=torch.load(mfile1, map_location=device))
        self.model2.load_state_dict(state_dict=torch.load(mfile2, map_location=device))
        # self.tl = llist

    

    def eval_on_val(self, val_loader):
        # val_loader: requires_index = True
        yt, yp = [], []
        self.model1.eval()
        self.model2.eval()
        with torch.no_grad():
            for xx1, xx2, yy in tqdm(val_loader):
                xx1,xx2, yy = xx1.to(self.device), xx2.to(self.device), yy
                zz = ((self.model1(xx1).detach().cpu()+self.model2(xx2).detach().cpu())/2 > 0.5).float().cpu()
                for y in yy: yt.append(y)
                for z in zz: yp.append(z)
            yt = torch.stack(yt, 0)
            yp = torch.stack(yp, 0)
            print(yp.shape)
            f1 = f1_score(yt,yp,average='samples')
            prec = precision_score(yt,yp,average='samples')
            reca = recall_score(yt,yp,average='samples')
            print(f'Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.5f}')
            
        return 




if __name__ == "__main__":
    d_path = './dataset/val_normd.json'
    model1 = Model1()
    model2 = Model_large()
    mfile2 = './output/base1/largebase1.ckpt'
    mfile1 = './output/base1/256base1.pt'
    val_ds = MyDataset(load_data(d_path))
    val_dl = DataLoader(val_ds,collate_fn=collate_fn_base_large,batch_size=256)
    predictor = Multi_label(model1,mfile1,model2,mfile2,torch.device('cuda'))
    predictor.eval_on_val(val_dl)

# # model.load_state_dict(torch.load(mfile), map_location=device)
# mfile1 = 'C:/Users/liuweican/人民网比赛/300base1cls.ckpt'
# model1 = Model()
# mfile2 = 'C:/Users/liuweican/人民网比赛/largebase1.ckpt'
# model2 = Model_large()

# ml = Multi_label(model1, mfile1, model2, mfile2, llist, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# x_l = sep_point([1, 2, 5, 100])
# print(x_l)

# val_data = load_data('C:/Users/liuweican/人民网比赛/val_normd.json')
# val_ds1 = MyDataset1(val_data, requires_index=True)
# val_ds2 = MyDataset2(val_data)
# val_dl1 = DataLoader(val_ds1, batch_size=32, collate_fn=collate_fn)
# val_dl2 = DataLoader(val_ds2, batch_size=32, collate_fn=collate_fn)
# bad_case_indices, res = ml.eval_on_val(val_dl1, val_dl2, x_l)
# print(res)

# fw = open('bad_case.json', 'w', encoding='utf-8')
# for i in bad_case_indices:
#     l = {}
#     l['text'] = val_data[i][0]
#     l['true_label'] = val_data[i][1]
#     l['pred_label'] = ml.predict(l['text'])[0]
#     l = json.dumps(l, ensure_ascii=False)
#     fw.write(l + '\n')
# fw.close()

# bad_case_indices = ml.eval_on_val(val_dl)
# print(len(bad_case_indices))
# text = val_data[0][0]
# print(text)
# print(ml.predict(text))
# print(val_data[0][1])