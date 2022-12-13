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
from base1 import sortlabel
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
    return [(x["text_normd"], x["ID"]) for x in LoadJsons(fn)]

class TestDS(Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = []        
        for d in data:
            text = d[0]
            id = d[1]
            id = torch.tensor([id])
            self.data.append([text,id])
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)

def collate_test(batch):
    z = tokenizer([d[0] for d in batch],return_tensors='pt',truncation=True, max_length=128,padding=True)
    return (z.input_ids,
            z.attention_mask,
            z.token_type_ids,
            torch.cat([x[1] for x in batch], 0))

class Inference:
    def __init__(self, model, mfile, llist, device):
        self.model = model.to(device)
        self.device = device
        self.model.load_state_dict(state_dict = torch.load(mfile, map_location=device))
        self.model.eval()
        # self.model4 = model4
        # self.model4.load_state_dict(state_dict = torch.load('output/base4/cls_base4.pt', map_location=device))
        self.tl = llist
        
    def infer_on_test(self,test_dl):
        # res: id to label list
        res = {}
        self.model.eval()
        with torch.no_grad():
            for xx,_,_,ids in tqdm(test_dl):
                xx = xx.to(self.device)
                scores = self.model(xx).detach().cpu()
                zz = (scores > 0.5).float().cpu()
                for idx,z in enumerate(zz): 
                    if z.sum() == 0:
                        pred_idx = scores[idx].argmax(-1)
                        res[int(ids[idx].item())] = [self.tl.get_token(pred_idx)]
                        #zz[idx] = torch.zeros(self.tl.get_num()).scatter(0,pred_idx,1)
                    else:
                        indices = z.nonzero().squeeze(1)
                        res[int(ids[idx].item())] = [self.tl.get_token(i) for i in indices]
        
        return res


def write_to_csv(id2l, out_path):
    data = {'ID':[], 'Label':[]}
    for k,v in id2l.items():
        data['ID'].append(str(k))
        data['Label'].append('，'.join(v))
    df = pd.DataFrame(data)
    df.to_csv(out_path, index=0,sep='\t')
    return



if __name__ == '__main__':
    transfer_test('./dataset/test.csv','./dataset/test.json')
    # model_name = 'hfl/chinese-roberta-wwm-ext'
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    test_data = load_test('./dataset/test.json')
    test_ds = TestDS(test_data)
    test_dl = DataLoader(test_ds,collate_fn=collate_test,batch_size=128)
    model = Model1()
    mfile = './output/base5/base5.ckpt'
    source = LoadJsons('./dataset/train.json')
    llist = sortlabel.TokenList('sortlabel.txt', source=source, func=lambda x:x['label'], low_freq=1, save_low_freq=1)
    Infer = Inference(model,mfile,llist,torch.device('cuda'))
    res = Infer.infer_on_test(test_dl)
    write_to_csv(res,'submission.csv')
    # print(list(res.values())[:5])
    # print(list(res.keys())[0])
