
import sys
sys.path.append('.')
# sys.path.append('./base4')
import torch
from model import Model1
from base4 import model
from main import tokenizer,llist,MyDataset,collate_fn,load_data
import numpy as np
from torch.utils.data import DataLoader
from sortlabel import sep_point,label2id,label2vec
import ljqpy
import json
from sklearn.metrics import *
from merge_func import Val0_Classify,Rules
from tqdm import tqdm
from base4.predict import get_embedding
 
def get_rank(tensor:torch.Tensor,ind):  # 要求的是某个index所在元素的排行
    t= tensor.sort(descending=True)[1].tolist()
    dic = {k:v+1 for v, k in enumerate(t)}
    # print(ind in t)
    return dic[ind]

class Multi_label:
    def __init__(self, model,mfile,llist, model4=None,mfile4=None,device=torch.device('cuda')):
        self.model = model.to(device)
        self.device = device
        self.model.load_state_dict(state_dict = torch.load(mfile, map_location=device))
        self.model.eval()
        if model4:
            self.model4 = model4
            self.model4.load_state_dict(state_dict = torch.load(mfile4, map_location=device))
            self.model4.eval()
            batch_size = 128
            self.rep = get_embedding(self.model4, 'dataset/train_normd.json', self.k, self.f, batch_size, self.device)
            self.model4.to(device)
            self.k = 3
            self.f = 15000
        self.tl = llist
 


    def eval_on_val(self,val_loader,x_l=None):
		# val_loader: requires_index = True,否则取不了bad_indices
        yt, yp, bad_case_indices, true_score_l, pred_score_l = [], [], [], [], []
        self.model.eval()
        pbar = tqdm(val_loader, total=len(val_loader))
        Classifier = Val0_Classify(llist)
        with torch.no_grad():
            for xx,__,__, yy,indices in pbar:
                xx = xx.to(self.device)
                scores = self.model(xx).detach().cpu()
                zz = (scores > 0.5).float().cpu()
                # true_id = yy.nonzero().squeeze(1)

                for idx,z in enumerate(zz):
                    true_id = yy[idx].nonzero().squeeze(1)
                    pred_id = z.nonzero().squeeze(1)
                    z_score = [(self.tl.get_token(j),scores[idx][j].item()) for j in pred_id]
                    if sum(z) == 0:
                        z_vec, z_score = Classifier.fun(scores[idx])
                        zz[idx] = z_vec
                    if any(yy[idx] != z): 
                        bad_case_indices.append(indices[idx])
                        pred_score_l.append(z_score)
                        true_score_l.append([(self.tl.get_token(i), scores[idx][i].item(), get_rank(scores[idx], i.item())) for i in true_id])
                    yt.append(yy[idx])
                    yp.append(zz[idx])

            yt = torch.stack(yt,0)
            yp = torch.stack(yp,0)
            print(yt.shape)
            accu = accuracy_score(yt,yp)
            prec = precision_score(yt,yp,average='samples',zero_division=0)
            reca = recall_score(yt,yp,average='samples',zero_division=0)
            f1 = f1_score(yt,yp,average='samples',zero_division=0)
            print(f'Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.5f}')

        return bad_case_indices,true_score_l, pred_score_l


def write_bad_to_json(val_data,bad_case_indices,true_score_l, pred_score_l, out_path):
    data = []
    for i in range(len(bad_case_indices)):
        d = {}
        d['text'] = val_data[bad_case_indices[i]][0]
        d['true_score'] = true_score_l[i]
        d['pred_score'] = pred_score_l[i]
        data.append(d)
    ljqpy.SaveJsons(data, out_path)


if __name__ == '__main__':
    val_data = load_data('./dataset/val_normd.json')
    val_ds = MyDataset(val_data,requires_index=True)
    val_dl = DataLoader(val_ds,collate_fn=collate_fn,batch_size=64)
    model = Model1()
    mfile = './output/base5/base5.ckpt'
    ml = Multi_label(model,mfile,llist)
    bad_case_indices,true_score_l, pred_score_l = ml.eval_on_val(val_dl)
    out_path = './output/result/bad_case.json'
    write_bad_to_json(val_data,bad_case_indices,true_score_l, pred_score_l, out_path)
