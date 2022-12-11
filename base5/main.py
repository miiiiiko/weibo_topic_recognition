import torch
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os
import sortlabel 
# from easonsi import utils
# from ljqpy import LoadJsons
# from ljqpy import TokenList
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import Model
import torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
import transformers
from tqdm import tqdm
import math
from sklearn import metrics
import time
import ljqpy
import pt_utils
from ljqpy import LoadJsons
from pairdata import StcPairSet,pairs_vs,sentence_pair_gen


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
model_name = 'hfl/chinese-roberta-wwm-ext'
tokenizer = BertTokenizer.from_pretrained(model_name)
source = ljqpy.LoadJsons('./dataset/train.json')
llist = sortlabel.TokenList('labellist.txt', source=source, func=lambda x:x['label'], low_freq=1, save_low_freq=1)


def load_data(fn):
    # print('loading data')
    return [(x["text_normd"], x["label"]) for x in ljqpy.LoadJsons(fn)]

# datadir = './dataset'

# xys = [load_data(os.path.join(datadir, '%s_normd.json') % tp) for tp in ['train', 'val']]



def label2vec(targrtlabels:list,dims= llist.get_num()):
    lab_vec = np.zeros(dims)
    
    for label in targrtlabels:
        loc = llist.get_id(label)
        if loc != -1:
            lab_vec[loc] = 1.0
    return lab_vec


class MyDataset(Dataset):
    def __init__(self,data,maxlen=128, requires_index = False):
        super().__init__()
        self.data = []        
        for i,d in enumerate(data):
            # print(d[0])
            text = tokenizer([d[0]],return_tensors='pt', truncation=True, max_length=maxlen)['input_ids'][0]
            # print(text)
            label = label2vec(d[1],dims= llist.get_num())
            label = torch.from_numpy(label)
            if requires_index:
                self.data.append([text,label,torch.tensor([i])])
            else:
                self.data.append([text,label])

    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


# x = ['今天天气真好','明天天气差','好饿']
# y = [[0,1,0],[1,0,1],[1,1,0]]
# data = zip(x,y)
# dataset = MyDataset(data)
# tokenizer = BertTokenizer(model_name)
def collate_fn(batch):
    text,label = [],[]
    for d in batch:
        text.extend(d[:2])
        label.extend([d[2],d[2]])
    z = tokenizer(text,return_tensors='pt',truncation=True, max_length=256,padding=True)
    return (z.input_ids,
            z.attention_mask,
            z.token_type_ids,
            torch.cat(label, 0)
    )


def cosent_loss(y_true,y_pred, lamb=20):
    # print(type(y_pred))
    # print(y_pred.shape)
    y_true = y_true[::2]
    #得到的y_true是(b,b)矩阵，第(i,j)元素仅当第i对句子标签0第j对句子标签1时是1，否则是0
    y_true = (y_true[:, None] < y_true[None, :]).float()
    y_pred = F.normalize(y_pred)
    y_pred = torch.sum(y_pred[::2] * y_pred[1::2], dim=1) * lamb
    # 得到的y_pred是（b,b）矩阵，第（i,j）元素是第i个句子对和第j个句子对的cosine相似度差值
    y_pred = y_pred[:, None] - y_pred[None, :]
    # 仅保存y_true=1即第i对句子0第j对句子1的y_pred
    y_pred = (y_pred - (1 - y_true) * 1e12).reshape(-1)
    y_pred = torch.cat((torch.tensor([0]).float().cuda(), y_pred), dim=0)
    return torch.logsumexp(y_pred,dim=0)





def plot_learning_curve(record,pic_n):
    # x1 = np.arange(config['total_steps'])
    # x2 = x1[::config['valid_steps']]
    y1 = record['train_loss']
    y2 = record['val_f1']
    # print(len(y1),len(y2))
    # y2 = [0 if np.isnan(i) else i for i in y2]
    x1 = np.arange(1,len(y1)+1)
    # print(int(len(y1)/len(y2)))
    x2 = x1[::int(len(y1)/len(y2))]
    # print(x2)

    fig = figure(figsize = (6,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(x1,y1, c = 'tab:red', label = 'train_loss')
    ax2 = ax1.twinx()
    ax2.plot(x2,y2, c='tab:cyan', label='val_f1')
    ax1.set_xlabel('steps')
    ax1.set_ylabel('train_loss')
    ax2.set_ylabel('val_f1')
    plt.title('Learning curve')
    ax1.legend(loc=1)
    ax2.legend(loc=2)
    # plt.show()
    plt.savefig(pic_n)
    return

def cal_hour(seconds):
    # seconds =35400
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    # print("%d:%02d:%02d" % (h, m, s))
    return "%d:%02d:%02d" % (h, m, s)
    

def train_model(model, optimizer, train_dpath, epochs=3, train_amount=100000, batch_size=8,train_func=None, test_func=None, 
                scheduler=None, save_file=None,accelerator=None, epoch_len=None):  # accelerator：适用于多卡的机器，epoch_len到该epoch提前停止
    best_f1 = -1
    for epoch in range(epochs):
        train_ds = StcPairSet(sentence_pair_gen,train_amount,0.3,train_dpath,20)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        model.train()
        print(f'\nEpoch {epoch+1} / {epochs}:')
        if accelerator:
            pbar = tqdm(train_dl, total=epoch_len, disable=not accelerator.is_local_main_process)
        else: 
            pbar = tqdm(train_dl, total=epoch_len)
        metricsums = {}
        iters, accloss = 0, 0
        for ditem in pbar:
            metrics = {}
            loss = train_func(model, ditem)
            if type(loss) is type({}):
                metrics = {k:v.detach().mean().item() for k,v in loss.items() if k != 'loss'}
                loss = loss['loss']
            iters += 1; accloss += loss
            optimizer.zero_grad()
            if accelerator: 
                accelerator.backward(loss)
            else: 
                loss.backward()
            optimizer.step()
            if scheduler:
                if accelerator is None or not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
            for k, v in metrics.items(): metricsums[k] = metricsums.get(k,0) + v
            infos = {'loss': f'{accloss/iters:.4f}'}
            for k, v in metricsums.items(): infos[k] = f'{v/iters:.4f}' 
            pbar.set_postfix(infos)
            if epoch_len and iters > epoch_len: break
        pbar.close()
        if test_func:
            if accelerator is None or accelerator.is_local_main_process: 
                model.eval()
                accu,prec,reca,f1 = test_func()
                if f1 >best_f1 and save_file:
                    if accelerator:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        accelerator.save(unwrapped_model.state_dict(), save_file)
                    else:
                        torch.save(model.state_dict(), save_file)
                    print(f"Epoch {epoch + 1}, best model saved. (Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.4f})")
                    best_f1 = f1


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = Model(model_name, llist.get_num()).to(device)
# mfile = '300base1.pt'


if __name__ == '__main__':
    record = {"train_loss":[],"val_f1":[]}
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(model_name).to(device)
    ds_test = pairs_vs
    # print('loading data completed')
    # dl_train = torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=32, collate_fn=collate_fn)
    # print("dataloader completed")
    # model = Model(model_name, llist.get_num()).to(device)
    print("finish loading model")
    mfile = './output/base4/cls_base4.pt'

    epochs = 30
    train_amount = 100000
    batch_size = 16
    total_steps = math.ceil(train_amount/batch_size)* epochs

    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, 1e-5, total_steps)
    loss_func = cosent_loss
    start_time = time.time()
    val_time = 0
    def train_func(model, ditem):
        input_ids,attention_mask,token_type_ids, yy = ditem[0].to(device), ditem[1].to(device),ditem[2].to(device),ditem[3].to(device)
        zz = model(input_ids,attention_mask,token_type_ids)
        loss = loss_func(yy, zz)
        record["train_loss"].append(loss.item())
        return {'loss': loss}

    def test_func(): 
        global val_time
        t1 = time.time()
        yt, yp = [], []
        model.eval()
        with torch.no_grad():
            for input_ids,attention_mask,token_type_ids, yy in dl_test:
                input_ids,attention_mask,token_type_ids = input_ids.to(device),attention_mask.to(device),token_type_ids.to(device)
                # xx, yy = xx.to(device), yy
                zz = model(input_ids,attention_mask,token_type_ids).detach().cpu()
                zz = F.normalize(zz, p=2, dim=1)
                zz = torch.sum(zz[::2] * zz[1::2], dim=1) 
                for y in yy[::2]: yt.append(y)
                for z in zz: yp.append((z>0.5)*1)
            # yt = torch.stack(yt,0).numpy().astype('int64')
            # yp = torch.stack(yp,0).numpy().astype('int64')
            accu = metrics.accuracy_score(yt,yp)
            prec = metrics.precision_score(yt,yp)
            reca = metrics.recall_score(yt,yp)
            f1 = metrics.f1_score(yt,yp)
            # accu = sum([all(yt[i] == yp[i])*1 for i in range(len(yt))])/len(yt)
            # # print(accu)
            # # print(type(accu))
            # # accu = (yt == yp).float().mean()
            # prec = (yt + yp > 1.5).float().sum() / max(yp.sum().item(),1)
            # reca = ((yt + yp > 1.5).float().sum() / max(yt.sum().item(),1))
            # # f1 = 2 * prec * reca / (prec + reca)
            # # f1 = 0 if np.isnan(f1) else f1.item()
            # #accu,prec,reca,f1 = accu.item(),prec.item(),reca.item(),f1.item()
            record["val_f1"].append(f1)
            # f1_1d = metrics.f1_score(yt.unsqueeze(1).numpy().astype('int64'),yp.unsqueeze(1).numpy().astype('int64'),average='samples')
        print(f'Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.5f}')
        model.train()
        t2 = time.time()
        val_time += t2-t1
        return accu,prec,reca,f1
    train_dpath = './dataset/train_normd.json'
    print('Start training!')
    # train_model(model, optimizer, '/home/qsm22/weibo_topic_recognition/dataset/train_normd.json',epochs,train_func, test_func,train_amount=train_amount,batch_size=batch_size,scheduler=scheduler, save_file=mfile)
    train_model(model, optimizer, train_dpath, epochs, train_amount, batch_size,train_func, test_func, scheduler, save_file=mfile)
    # plot_learning_curve(record)
    end_time = time.time()
    val_time = val_time/epochs
    total_time = end_time-start_time
    total_time = total_time/epochs
    train_time = total_time - val_time
    total_time,train_time,val_time = cal_hour(total_time),cal_hour(train_time),cal_hour(val_time)
    print(f'Train_time:{train_time}, Val_time:{val_time}, total_time:{total_time}')
    plot_learning_curve(record,'./output/base4/base4_learn_curve')
    print('done')
    
# else:
#     model.load_state_dict(torch.load(mfile), map_location=device)
#     # model.eval()