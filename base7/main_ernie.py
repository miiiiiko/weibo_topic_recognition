import sys
sys.path.append(".")
import torch
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os
from base1 import sortlabel,ljqpy, pt_utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model7 import Model7
import torch.nn as nn
import transformers
from tqdm import tqdm
import torch.nn.functional as F
from sklearn import metrics
import time
import emojiswitch
# from ljqpy import LoadJsons

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
model_name = "nghuyong/ernie-3.0-base-zh"
# model_name_l = 'hfl/chinese-roberta-wwm-ext-large'
tokenizer = BertTokenizer.from_pretrained(model_name)
source = ljqpy.LoadJsons('./dataset/train.json')
llist = sortlabel.TokenList('sortlabel.txt', source=source, func=lambda x:x['label'], low_freq=1, save_low_freq=1)

cc = {'𝓪':'a','𝒶':'a','𝒜':'A','𝓐':'A','𝒂':'a','ⓐ':'a','𝐴':'A','𝑎':'a','𝗮':'a','𝗔':'A','𝟬':'0'}
fconv = {}
for x, y in cc.items():
    mx = 10 if y == '0' else 26
    for i in range(mx): 
        fconv[chr(ord(x)+i)] = chr(ord(y)+i)
def ConvertFlower(zz):
    newz = []
    for z in zz: newz.append(fconv.get(z, z))
    return ''.join(newz)

def Normalize(z):
    z = ConvertFlower(z)
    return emojiswitch.demojize(z, delimiters=('#',''), lang="zh")


def load_data(fn):
    # print('loading data')
    return [(Normalize(x["text_normd"]), x["label"]) for x in ljqpy.LoadJsons(fn)]
datadir = './dataset'

xys = [load_data(os.path.join(datadir, '%s_normd.json') % tp) for tp in ['train', 'val']]


def label2vec(targrtlabels:list,dims= llist.get_num()):
    lab_vec = np.zeros(dims)
    
    for label in targrtlabels:
        loc = llist.get_id(label)
        if loc != -1:
            lab_vec[loc] = 1.0
    return lab_vec


class MyDataset(Dataset):
    def __init__(self,data, requires_index = False):
        super().__init__()
        self.data = []        
        for i,d in enumerate(data):
            # print(d[0])
            text = d[0]
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

def collate_fn(batch):
    # print(len(batch))
    z = tokenizer([d[0] for d in batch],return_tensors='pt',truncation=True, max_length=128,padding=True)
    if len(batch[0]) ==2:
        return (z.input_ids,
            z.attention_mask,
            z.token_type_ids,
            torch.stack([x[1] for x in batch], 0))
    else:
        return (z.input_ids,
            z.attention_mask,
            z.token_type_ids,
            torch.stack([x[1] for x in batch], 0),
            torch.cat([x[2] for x in batch]))
        
# (b, max_len)

def plot_learning_curve(record,pic_n):
    y1 = record['train_loss']
    y2 = record['val_f1']
    x1 = np.arange(1,len(y1)+1)
    x2 = x1[::int(len(y1)/len(y2))]
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
    return "%d:%02d:%02d" % (h, m, s)


def compute_kl_loss(p, q, pad_mask=None):
    # pad_mask = (pad_mask > 0.5)
    # p_plus = torch.stack([p,1-p],2)
    # q_plus = torch.stack([q,1-q],2)
    p_loss = F.kl_div(F.log_softmax(p,dim=-1), F.softmax(q,dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q,dim=-1), F.softmax(p,dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss

def multilabel_categorical_crossentropy(y_pred,y_true):
    y_pred = (1 - 2 * y_true) * y_pred    # y_true为0的项，y_pred不变，否则×-1
    y_pred_neg = y_pred - y_true * 1e12   # y_true为1的项，y_pred变成-无穷，否则不变
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # y_true为0的项，y_pred变负无穷，否则变为原来-1
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1).mean()
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1).mean()
    return neg_loss + pos_loss


def train_model(model, optimizer, train_dl, epochs=3, train_func=None, test_func=None, 
                scheduler=None, save_file=None, accelerator=None, epoch_len=None):  # accelerator：适用于多卡的机器，epoch_len到该epoch提前停止
    best_f1 = -1
    for epoch in range(epochs):
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
                if f1 >=best_f1 and save_file:
                    if accelerator:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        accelerator.save(unwrapped_model.state_dict(), save_file)
                    else:
                        torch.save(model.state_dict(), save_file)
                    print(f"Epoch {epoch + 1}, best model saved. (Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.4f})")
                    best_f1 = f1



if __name__ == '__main__':
    record = {"train_loss":[],"val_f1":[]}
    # device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(llist.get_num())
    # print("cuda" if torch.cuda.is_available() else "cpu")
    model = Model7(model_name, llist.get_num(),ernie=True).to(device)
    ds_train, ds_test = MyDataset(xys[0]), MyDataset(xys[1])
    print('loading data completed')
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=32, collate_fn=collate_fn)
    print("dataloader completed")
  
    print("finish loading model")
    mfile = './output/base7/128_base7.ckpt'
    # model.load_state_dict(state_dict = torch.load(mfile, map_location=device))

    alpha = 4
    epochs = 40
    total_steps = len(dl_train) * epochs

    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, 5e-5, total_steps)
    loss_func = multilabel_categorical_crossentropy
    start_time = time.time()
    val_time = 0
    def train_func(model, ditem):
        xx, yy = ditem[0].to(device), ditem[-1].to(device)
        zz1 = model(xx)
        zz2 = model(xx)

        multi_loss = 0.5*(loss_func(zz1.float(), yy.float()) + loss_func(zz2.float(), yy.float()))
        kl_loss = compute_kl_loss(zz1, zz2)
        loss = multi_loss + alpha * kl_loss

        record["train_loss"].append(loss.item())
        return {'loss': loss}

    def test_func(): 
        global val_time
        t1 = time.time()
        yt, yp = [], []
        model.eval()
        with torch.no_grad():
            for xx,_,_, yy in dl_test:
                xx, yy = xx.to(device), yy
                zz = (model(xx).detach().cpu() > 0).float().cpu()
                for y in yy: yt.append(y)
                for z in zz: yp.append(z)
            yt = torch.stack(yt,0).numpy().astype('int64')
            yp = torch.stack(yp,0).numpy().astype('int64')
            accu = metrics.accuracy_score(yt,yp)
            prec = metrics.precision_score(yt,yp,average='samples',zero_division=0)
            reca = metrics.recall_score(yt,yp,average='samples',zero_division=0)
            f1 = metrics.f1_score(yt,yp,average='samples',zero_division=0)
            
            record["val_f1"].append(f1)
            # f1_1d = metrics.f1_score(yt.unsqueeze(1).numpy().astype('int64'),yp.unsqueeze(1).numpy().astype('int64'),average='samples')
        print(f'Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.5f}')
        model.train()
        t2 = time.time()
        val_time += t2-t1
        return accu,prec,reca,f1

    print('Start training!')
    train_model(model, optimizer, dl_train, epochs, train_func, test_func, scheduler=scheduler, save_file=mfile)
    # plot_learning_curve(record)
    end_time = time.time()
    val_time = val_time/epochs
    total_time = end_time-start_time
    total_time = total_time/epochs
    train_time = total_time - val_time
    total_time,train_time,val_time = cal_hour(total_time),cal_hour(train_time),cal_hour(val_time)
    print(f'Train_time:{train_time}, Val_time:{val_time}, total_time:{total_time}')
    plot_learning_curve(record,'./output/base7/128base7_rdrop')
    print('done')
    
# else:
#     model.load_state_dict(torch.load(mfile), map_location=device)
#     # model.eval()