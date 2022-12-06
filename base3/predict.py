from main import load_data,DataLoader,collate_fn,llist,tokenizer,label2vec
from sklearn import metrics
from model import Model
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

label2text = defaultdict(list) 
train_data = load_data('./dataset/train_normd.json')
# train_ds = MyDataset(train_data,requires_index=True)
# choose text from train_ds

for d in train_data:
    for l in d[1]:
        label2text[llist.get_id(l)].append(d[0])

# print([len(l) for l  in label2text.values()])

def select_text(k, label2text):
    for i,text_l in label2text.items():
        if len(text_l)> k:
            label2text[i] = random.sample(text_l,k=k)
    return label2text

# print(len(label2text))
# print([len(l) for l in select_text(2,label2text).values()])
label2text = select_text(3,label2text)

# (TEXT,LABEL)
class val_ds(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = []
        for d in data:
            text = d[0]
            label = label2vec(d[1],dims= llist.get_num())
            label = torch.from_numpy(label)
            self.data.append([text,label])

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

class Text_sim:
    def __init__(self, model,mfile,llist,label2text,device):
        self.model = model.to(device)
        self.device = device
        self.model.load_state_dict(state_dict = torch.load(mfile, map_location=device))
        self.model.eval()
        self.tl = llist
        self.label2text = label2text
        

    # 对一个句子预测标签,仅预测n以后的标签
    def predict(self, sent, threshold=0.9,n=0):
        res = [0] * llist.get_num()
        for i in range(llist.get_num()):
            text_l = label2text[i]
            if i < n or not text_l:
                continue
            input = [[sent,t] for t in text_l]
            # print(input)
        # print(len(input))
        with torch.no_grad():
            zz = tokenizer(input,return_tensors='pt',truncation=True, max_length=512,padding=True)
            input_ids = zz.input_ids.to(self.device)
            attention_mask = zz.attention_mask.to(self.device)
            token_type_ids = zz.token_type_ids.to(self.device)
            # input_ids,attention_mask,token_type_ids = input_ids.to(self.device),attention_mask.to(self.device),token_type_ids.to(self.device)
                    # print(F.softmax(self.model(input_ids,attention_mask,token_type_ids).detach().cpu()))
            dist = F.softmax(self.model(input_ids,attention_mask,token_type_ids).detach(),dim=1).cpu()
            scores = dist[:,1]
            if scores.mean() > threshold:
                res[i] = 1
            return res,scores
        #     scores = scores.reshape(-1,3)
        #     indices =  ((scores.mean(1) > threshold)*1).nonzero()
        #     for i in indices:
        #         res[i] = 1
        # return res


    def eval_on_val(self,val_data,threshold=0.1,n=800):
        self.model.eval()
        yt,yp = [],[] 
        for d in val_data:
            yt.append(label2vec(d[1]).astype('int64').tolist())
            # print(d[0])
            yp.append(self.predict(d[0],threshold)[0])
        if not n:
            yt,yp = yt[n:],yp[n:]
        f1 = metrics.f1_score(yt,yp,average='samples')
        return f1

        # with torch.no_grad():
        #     for input_ids,attention_mask,token_type_ids, yy in val_dl:
        #         self.model()
        # for i in range(val_data):

        # return
            


mfile = '/home/qsm22/weibo_topic_recognition/512_base3.pt'
model = Model()
ts = Text_sim(model,mfile,llist,label2text,torch.device("cuda" if torch.cuda.is_available() else "cpu"))

val_data = load_data('/home/qsm22/weibo_topic_recognition/dataset/test_normd.json')
# for d in val_data:
#     print(d[0])
#     print(d[1])
#     break
print(ts.eval_on_val(val_data))
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