
import sys
sys.path.append('.')
from transformers import BertTokenizer
# sys.path.append('./base4')
import torch
import torch.nn.functional as F
from model import Model1
from base4 import model
from base1.main import label2vec
from base7 import model7
from main import tokenizer,llist,MyDataset,collate_fn,load_data
from base7.model7 import Model7
# from base7.main_ernie import collate_fn
from base7.main7 import load_data2
import numpy as np
from torch.utils.data import DataLoader,Dataset
from sortlabel import sep_point,label2id
import ljqpy
import json
from sklearn.metrics import *
from merge_func import Val0_Classify,Rules
from tqdm import tqdm
from base4.predict import get_embedding
from copy import deepcopy 
import matplotlib.pyplot as plt
from pred_helper import transfer,transfer_z,check_text

tokenizer_ernie = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")

# 现在有两种输入，去除表情版本和不去表情版, 重新定义collate_fn与load_data,dataset
def merge_data(dpath):
    ret = []
    data1 = load_data(dpath)
    data2 = load_data2(dpath)
    for i in range(len(data1)):
        ret.append((data1[i][0],data2[i][0],data1[i][1]))
    return ret


class MergeDataset(Dataset):
    def __init__(self,data, requires_index = False):
        super().__init__()
        self.data = []        
        for i,d in enumerate(data):
            text1 = d[0]
            text2 = d[1]
            label = label2vec(d[2])
            label = torch.from_numpy(label)
            if requires_index:
                self.data.append([text1,text2,label,torch.tensor([i])])
            else:
                self.data.append([text1,text2,label])

    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def merge_fn(batch):
    # print(len(batch))
    z1 = tokenizer([d[0] for d in batch],return_tensors='pt',truncation=True, max_length=128,padding=True)
    z2 = tokenizer([d[1] for d in batch],return_tensors='pt',truncation=True, max_length=128,padding=True)
    # z3 = tokenizer([d[1] for d in batch],return_tensors='pt',truncation=True, max_length=128,padding=True)
    if len(batch[0]) ==3:
        return (z1.input_ids,
            z2.input_ids,
            # z3.input_ids,
            torch.stack([x[2] for x in batch], 0))
    else:
        return (z1.input_ids,
            z2.input_ids,
            # z3.input_ids,
            torch.stack([x[2] for x in batch], 0),
            torch.cat([x[3] for x in batch]))

def get_rank(tensor:torch.Tensor,ind):  # 要求的是某个index所在元素的排行
    t= tensor.sort(descending=True)[1].tolist()
    dic = {k:v+1 for v, k in enumerate(t)}
    # print(ind in t)
    return dic[ind]

def pick_len(data, l=10):
    res = []
    for d in data:
        if len(d[0]) <= l:
            res.append(d)
    return res

# f1_l = []
zero_indices = []
class Multi_label:
    def __init__(self, model,mfile_l,llist,n=4,device=torch.device('cuda')):
        # self.model = model.to(device)
        self.device = device
        self.model_l = [deepcopy(model).to(device) for i in range(len(mfile_l))]
        for i,mfile in enumerate(mfile_l):
            # print(i)
            self.model_l[i].load_state_dict(state_dict = torch.load(mfile, map_location=device))
            self.model_l[i].eval()
        self.tl =llist
        self.n = n # 从n开始是去表情版


    def eval_on_val(self,val_data,val_loader):
		# val_loader: requires_index = True,否则取不了bad_indices
        yt, yp, bad_case_indices, true_score_l, pred_score_l = [], [], [], [], []
        # w = torch.tensor([0.2]*5)
        # self.model.eval()
        pbar = tqdm(val_loader, total=len(val_loader))
        Classifier = Val0_Classify(llist)
        with torch.no_grad():
            for xx1,xx2, yy,indices in pbar:
                xx1,xx2 = xx1.to(self.device),xx2.to(self.device)
                scores_l = torch.zeros(yy.shape[0],yy.shape[1],len(self.model_l))
                for i,model in enumerate(self.model_l):
                    if i < self.n:
                        scores_l[...,i] = model(xx1).detach().cpu()
                    # elif i == len(self.model_l)-1:
                    #     scores_l[...,i] = model(xx3).detach().cpu()
                    else:
                        scores_l[...,i] = model(xx2).detach().cpu()
                    # if i > 1:
                    # scores_l[...,i] = torch.sigmoid(scores_l[...,i])

                # print(scores_l[...,0]==scores_l[...,1])

                scores = scores_l.mean(-1)
                # scores = torch.matmul(scores_l,w)
                scores_max,_ = scores_l.max(-1)
                # scores_l_vote = (scores_l > 0.5).float().cpu()
                # zz1 = (scores_l_vote.mean(-1)> 0.5).float().cpu()
                scores = transfer(scores)
                zz = (scores > 0.5).float().cpu()
                # zz = ((zz1 + zz2) > 1).float().cpu()
                # true_id = yy.nonzero().squeeze(1)
                # scores = transfer(scores)
                # scores_max = transfer(scores_max)


                for idx,z in enumerate(zz):
                    text = val_data[indices[idx]][0]
                    ind_plus = check_text(text)
                    for j in ind_plus:
                        z[j] = 1
                    z_vec = z
                    # true_id = yy[idx].nonzero().squeeze(1)
                    # pred_id = z.nonzero().squeeze(1)
                    # z_score = [(self.tl.get_token(j),scores[idx][j].item()) for j in pred_id]
                    if sum(z) == 0:
                        z_vec = Classifier.fun(scores_max[idx])
                    # z_vec = transfer_z(z_vec)
                    zz[idx] = z_vec
                    # if any(yy[idx] != zz[idx]): 
                    #     bad_case_indices.append(indices[idx])
                    #     pred_score_l.append(z_score)
                    #     true_score_l.append([(self.tl.get_token(i), scores[idx][i].item(), get_rank(scores[idx], i.item())) for i in true_id])
                    # if zz[idx][78]==1 or zz[idx][79]==1 or scores[idx][78] + scores[idx][79] > 0.5:
                    #     zz[idx][78],zz[idx][79] = 1,1 
                    # zz[idx] = transfer_z(zz[idx], scores_max[idx])
                    yt.append(yy[idx])
                    yp.append(zz[idx])

            yt = torch.stack(yt,0)
            yp = torch.stack(yp,0)
            # for i,d in enumerate(val_data):
            #     text = d[0]
            #     ind_plus = check_text(text)
            #     for j in ind_plus:
            #         print(yp[i][j])
            #         yp[i][j] = 1
            # print(yt.shape)
            for i in range(len(yt[0])):
                yt_i, yp_i = yt[:,i],yp[:,i]
                if all(yp_i==0):
                    continue
                if f1_score(yt_i,yp_i,zero_division=0) == 0:
                    zero_indices.append(i)
            accu = accuracy_score(yt,yp)
            prec = precision_score(yt,yp,average='samples',zero_division=0)
            reca = recall_score(yt,yp,average='samples',zero_division=0)
            f1 = f1_score(yt,yp,average='samples',zero_division=0)
            print(f'Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.5f}')
            print(zero_indices)
            # plt.savefig('./output/result/merge_plot')

        return bad_case_indices #,true_score_l, pred_score_l


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
    val_data = merge_data('./dataset/val_normd2.json')
    # val_data = pick_len(val_data,50)
    # print(len(val_data))
    val_ds = MergeDataset(val_data,requires_index=True)
    val_dl = DataLoader(val_ds,collate_fn=merge_fn,batch_size=128)
    model = Model1()
    model2 = Model7(ernie = True)
    # './output/base1/256base1.pt',
    mfile_l =  ['./output/base5/base5.ckpt', './output/base1/256base1.pt','./output/base7/128_base7.ckpt', './output/extra/128_base7_plus5.ckpt','./output/base7/wb_base_noseg_normd1_8855.pt',
    './output/ljq/wb_base_retrain_lock7_diceloss2_normd1_8846.pt','./output/ljq/wb_base_retrain2_lock10_diceloss_normd1_8871.pt','./output/ljq/wb_base_retrain2_lock7_diceloss_normd1_8858.pt',
    './output/ljq/wb_base_retrain_lock8_normd1_8844.pt','./output/ljq/wb_base_retrain_lock7_diceloss_normd1_8867.pt','./output/ljq/wb_base_retrain_lock6_normd1_8859.pt',
    './output/base7/base7noemo.ckpt','./output/base7/base7noemo_n2.ckpt']
    mfile_l2 = ['./output/ljq/wb_base_retrain_lock7_diceloss_normd2_8981.pt','./output/ljq/wb_base2_noseg_normd2.pt']+mfile_l
    # mfile_l = ['./output/base5/base5.ckpt','./output/base7/128_base7.ckpt', './output/extra/128_base7_plus5.ckpt','./output/base7/wb_base_noseg_normd1_8855.pt',
    # './output/ljq/wb_base_retrain_lock8_normd1_8844.pt','./output/ljq/wb_base_retrain_lock7_diceloss_normd1_8867.pt','./output/ljq/wb_base_retrain_lock6_normd1_8859.pt',
    # './output/base7/base7noemo.ckpt']
    # mfile_l = ['/home/qsm22/weibo_topic/output/base7/base7ernie.ckpt'] 
    ml = Multi_label(model,mfile_l,llist,n=13)
    ml.eval_on_val(val_data,val_dl)
    # bad_case_indices,true_score_l, pred_score_l = ml.eval_on_val(val_dl)
    # out_path = './output/result/bad_case.json'
    # write_bad_to_json(val_data,bad_case_indices,true_score_l, pred_score_l, out_path)
