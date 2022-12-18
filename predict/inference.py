import sys
sys.path.append('.')
sys.path.append('./base1')
import pandas as pd
import unicodedata
from base1.ljqpy import SaveJsons,LoadJsons
from transformers import BertTokenizer
from torch.utils.data import DataLoader,Dataset
import torch
from tqdm import tqdm
from base1.model import Model1
from base1 import sortlabel,pred_helper
from base1.mergepredict import merge_data,MergeDataset,merge_fn
from base7.main7 import Normalize
from copy import deepcopy
from base1.merge_func import Val0_Classify
# from base1.merge_func import Val0_Classify
# 采用目前最好的指标，多分类+argmax
model_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(model_name)
# 测试集要求id和label
def transfer_test(dpath,outpath):
    data = []
    df = pd.read_csv(dpath,  sep='\t', encoding="utf-8")
    for i in range(len(df)):
        l = {}
        l['text_normd'] = df.iloc[i,1].replace('\u200b','')
        l['text_normd'] =  unicodedata.normalize('NFKC', l['text_normd'])
        l['ID'] = int(df.iloc[i,0])
        data.append(l)
    SaveJsons(data,outpath)
    return 

def load_test(fn):
    return [(x["text_normd"],Normalize(x["text_normd"]), x["ID"]) for x in LoadJsons(fn)]

class TestDS(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = []        
        for i,d in enumerate(data):
            text1 = d[0]
            text2 = d[1]
            id = d[2]
            id = torch.tensor([id])
            i = torch.tensor([i])
            self.data.append([text1,text2,id,i])
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def collate_test(batch):
    z1 = tokenizer([d[0] for d in batch],return_tensors='pt',truncation=True, max_length=128,padding=True)
    z2 = tokenizer([d[1] for d in batch],return_tensors='pt',truncation=True, max_length=128,padding=True)
    return (z1.input_ids,
            z2.input_ids,
            torch.cat([x[2] for x in batch], 0),
            torch.cat([x[3] for x in batch], 0))

class Inference:
    def __init__(self, model, mfile_l, llist, device=torch.device('cuda'),n=11):
        self.device = device
        self.model_l = [deepcopy(model).to(device) for i in range(len(mfile_l))]
        for i,mfile in enumerate(mfile_l):
            self.model_l[i].load_state_dict(state_dict = torch.load(mfile, map_location=device))
            self.model_l[i].eval()
        self.tl = llist
        self.n = n
        
    def infer_on_test(self,test_data,test_dl):
        # res: id to label list
        res = {}
        res2 = []
        Classifier = Val0_Classify(llist)
        with torch.no_grad():
            for xx1,xx2,ids,indices in tqdm(test_dl):
                # print(indices)
                xx1,xx2 = xx1.to(self.device),xx2.to(self.device)
                scores_l = torch.zeros(xx1.shape[0],self.tl.get_num(),len(self.model_l))
                # print(scores_l.shape)
                for i,model in enumerate(self.model_l):
                    if i < self.n:
                        scores_l[...,i] = model(xx1).detach().cpu()
                    else:
                        scores_l[...,i] = model(xx2).detach().cpu()
                scores = scores_l.mean(-1)
                scores = pred_helper.transfer(scores)
                zz = (scores > 0.5).float().cpu()
                # xx = xx.to(self.device)
                # scores = self.model(xx).detach().cpu()
                # zz = (scores > 0.5).float().cpu()
                for idx,z in enumerate(zz):
                    text = test_data[indices[idx]][0]
                    ind_plus = pred_helper.check_text(text)
                    for j in ind_plus:
                        z[j] = 1
                    z_vec = z 
                    if sum(z) == 0:
                        z_vec = Classifier.fun(scores[idx])
                    # if z.sum() == 0:
                    #     score, ii = torch.sort(scores[idx],descending=True)
                    #     if score[1] > 0.5*score[0]:
                    #         pred_idx = ii[:2]
                    #     else:
                    #         pred_idx = ii[0].unsqueeze(0)
                    # #     # pred_idx = scores[idx].argmax(-1)
                    #     res[int(ids[idx].item())] = [self.tl.get_token(i) for i in pred_idx]
                    #     #zz[idx] = torch.zeros(self.tl.get_num()).scatter(0,pred_idx,1)
                    # else:
                    iis = z_vec.nonzero().squeeze(1)
                    res[int(ids[idx].item())] = [self.tl.get_token(i) for i in iis]
                    res2.append({'text':text,'pred':[(self.tl.get_token(i),scores[idx][i].item()) for i in iis]})
        SaveJsons(res2, './output/result/infer_score_1.json')
        return res


def write_to_csv(id2l, out_path):
    data = {'ID':[], 'Label':[]}
    for k,v in id2l.items():
        data['ID'].append(str(k))
        data['Label'].append('，'.join(v))
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=0,sep='\t')
    return

# def write_score_to_json(res2,outpath):
#     data = []
#     for i in range(len(bad_case_indices)):
#         d = {}
#         d['text'] = val_data[bad_case_indices[i]][0]
#         d['true_score'] = true_score_l[i]
#         d['pred_score'] = pred_score_l[i]
#         data.append(d)
#     ljqpy.SaveJsons(data, out_path)

if __name__ == '__main__':
    transfer_test('./dataset/test.csv','./dataset/test.json')
    # model_name = 'hfl/chinese-roberta-wwm-ext'
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    test_data = load_test('./dataset/test.json')
    test_ds = TestDS(test_data)
    test_dl = DataLoader(test_ds,collate_fn=collate_test,batch_size=128)
    model = Model1()
    mfile_l =  ['./output/ljq/wb_base_retrain_lock7_diceloss_normd2_8981.pt','./output/ljq/wb_base2_noseg_normd2.pt'] + ['./output/base5/base5.ckpt', './output/base1/256base1.pt','./output/base7/128_base7.ckpt', './output/extra/128_base7_plus5.ckpt','./output/base7/wb_base_noseg_normd1_8855.pt',
    './output/ljq/wb_base_retrain_lock7_diceloss2_normd1_8846.pt','./output/ljq/wb_base_retrain2_lock10_diceloss_normd1_8871.pt','./output/ljq/wb_base_retrain2_lock7_diceloss_normd1_8858.pt',
    './output/ljq/wb_base_retrain_lock8_normd1_8844.pt','./output/ljq/wb_base_retrain_lock7_diceloss_normd1_8867.pt','./output/ljq/wb_base_retrain_lock6_normd1_8859.pt',
    './output/base7/base7noemo.ckpt','./output/base7/base7noemo_n2.ckpt']
    # print(len(mfile_l))
    source = LoadJsons('./dataset/train.json')
    llist = sortlabel.TokenList('sortlabel.txt', source=source, func=lambda x:x['label'], low_freq=1, save_low_freq=1)
    Infer = Inference(model,mfile_l,llist,torch.device('cuda'),n=13)
    res = Infer.infer_on_test(test_data,test_dl)
    write_to_csv(res,'submission1.csv')
    # print(list(res.values())[:5])
    # print(list(res.keys())[0])
