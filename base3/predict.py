from main import load_data,DataLoader,collate_fn,llist,label2vec, cal_hour
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

label2text = defaultdict(list) 
label2text[0] = []
train_data = load_data('./dataset/train_normd.json')
# train_ds = MyDataset(train_data,requires_index=True)
# choose text from train_ds

for d in train_data:
    for l in d[1]:
        label2text[llist.get_id(l)].append(d[0])

# print([len(l) for l  in label2text.values()])

def select_text(k, label2text):
    for i,text_l in label2text.items():
        if text_l:
            label2text[i] = random.choices(text_l,k=k)
    return label2text

# print(len(label2text))
# print([len(l) for l in select_text(2,label2text).values()])
label2text = select_text(1,label2text)
# print(label2text[0])
# text_num =[]
# for i in range(len(label2text)):
#     text_num.append(len(label2text[i]))
# print(text_num)
# (TEXT,LABEL)

# for each text
class Text_DS(Dataset):
    def __init__(self,text,label2text,n=0):
        super().__init__()
        self.data = []
        # print(len(label2text))
        for i in range(n,len(label2text)):
            text_l = label2text[i]
            for t in text_l:
                self.data.append([t,text,torch.tensor([i])])


    def __getitem__(self, index):
        # [text1，text2, 索引]
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

# ds = Text_DS('你好',label2text)
# print(len(ds))

class Text_sim:
    def __init__(self, model,mfile,llist,label2text,device):
        self.model = model.to(device)
        self.device = device
        self.model.load_state_dict(state_dict = torch.load(mfile, map_location=device))
        self.model.eval()
        self.tl = llist
        self.label2text = label2text
        

    # 对一个句子预测标签
    def predict(self, sent, threshold=0.5,f=20):
        n = loc(f,llist)
        scores_sum = [0] * self.tl.get_num()
        # indices_count = [0] * llist.get_num()
        # total_count = [0,0]
        text_ds = Text_DS(sent,self.label2text,n)
        # print(n,len(text_ds))
        # print(len(text_ds))
        text_dl = DataLoader(text_ds,collate_fn=collate_fn,batch_size=32)
        print(len(text_ds),n)
        for ditem in text_dl:
            with torch.no_grad():
                # total_count[0] += 1
                input_ids,attention_mask,token_type_ids, indices = ditem[0].to(self.device), ditem[1].to(self.device),ditem[2].to(self.device),ditem[3]
                zz = self.model(input_ids,attention_mask,token_type_ids).detach()
                scores = F.softmax(zz,dim=1).cpu()[:,1]
                for i in range(len(indices)):
                    scores_sum[indices[i]] += scores[i]
                    # indices_count[indices[i]] += 1
                    # total_count[1] += 1
        res = [1 if item > threshold else 0 for item in scores_sum]
        return res,n
        # for i in range(llist.get_num()):
        #     text_l = label2text[i]
        #     if i < n or not text_l:
        #         continue
        #     input = [[sent,t] for t in text_l]
        #     # print(input)
        #     # print(len(input))
        #     with torch.no_grad():
        #         zz = tokenizer(input,return_tensors='pt',truncation=True, max_length=512,padding=True)
        #         input_ids = zz.input_ids.to(self.device)
        #         attention_mask = zz.attention_mask.to(self.device)
        #         token_type_ids = zz.token_type_ids.to(self.device)
        #         # input_ids,attention_mask,token_type_ids = input_ids.to(self.device),attention_mask.to(self.device),token_type_ids.to(self.device)
        #                 # print(F.softmax(self.model(input_ids,attention_mask,token_type_ids).detach().cpu()))
        #         dist = F.softmax(self.model(input_ids,attention_mask,token_type_ids).detach(),dim=1).cpu()
        #         # print(dist.shape)
        #         scores = dist[:,1]
        #         # print(i,scores)
        #         if scores.mean() > threshold:
        #             res[i] = 1
        # return res
        

    def eval_on_val(self,val_data,threshold=0.7,f=20):
        pbar = tqdm(val_data, total=len(val_data))
        self.model.eval()
        yt,yp = [],[] 
        # scores = []
        for d in pbar:
            # print(i)
            # print(d)
            yt.append(label2vec(d[1]).astype('int64').tolist())
            # print(d[0])
            res,n = self.predict(d[0],threshold=threshold,f=f)
            yp.append(res)
            #infos = {'index':{i}}
            #pbar.set_postfix(infos)
            # scores.append(score)
        # if not n:
        #     yt,yp = yt[n:],yp[n:]
        # print(n)
        yt,yp = np.array(yt),np.array(yp)
        yt,yp = yt[:,n:],yp[:,n:]
        pbar.close()
        f1 = metrics.f1_score(yt,yp,average='samples')
        macro_low_freq_f1 = metrics.f1_score(yt,yp,average='macro')
        micro_low_freq_f1 = metrics.f1_score(yt,yp,average='micro')
        prec = metrics.precision_score(yt,yp,average='micro')
        reca = metrics.recall_score(yt,yp,average='micro')
        print(f"Macro_f1: {macro_low_freq_f1 :.4f},  Micro_f1: { micro_low_freq_f1:.4f}, example_f1: {f1:.4f}, micro_prec{prec:.4f}, micro_rec{reca:.4f}")
        return 

        # with torch.no_grad():
        #     for input_ids,attention_mask,token_type_ids, yy in val_dl:
        #         self.model()
        # for i in range(val_data):

        # return
            


mfile = '/home/qsm22/weibo_topic_recognition/512_base3.pt'
model = Model()
ts = Text_sim(model,mfile,llist,label2text,torch.device("cuda" if torch.cuda.is_available() else "cpu"))
infer_start = time.time()
val_data = load_data('/home/qsm22/weibo_topic_recognition/dataset/val_normd.json')[:2]
print('val_data:')
print(val_data)
# print(ts.predict(val_data[0][0]))
ts.eval_on_val(val_data,threshold=0.1,f=5)
infer_end = time.time()
print('Inference time per text')
print(cal_hour((infer_end-infer_start)/len(val_data)))
# # for d in val_data:
#     print(d[0])
#     print(d[1])
#     break
#print(ts.predict(val_data[0][0]))
# val_ds = MyDataset(val_data,requires_index=True)
# val_dl = DataLoader(val_ds,batch_size=32,collate_fn=collate_fn)
# bad_case_indices, true_f1, res = ml.eval_on_val(val_dl,x_l)
# print(f'example_f1: {true_f1:.4f}')
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