import numpy as np
import matplotlib.pyplot as plt
import os, time,torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
from sortlabel import label2vec,TokenList
from ljqpy import LoadJsons
from model import Model_large,Model
import pt_utils
from sklearn.metrics import *

train_time = val_time = 0
def load_data(d_path):
    return [(x['text_normd'],x['label']) for x in LoadJsons(d_path)]
datadir = './'
xys = [load_data(os.path.join(datadir, '%s_normd.json') % tp) for tp in ['temp', 'temp']]
model_name = 'hfl/chinese-roberta-wwm-ext-large'
tokenizer = BertTokenizer.from_pretrained(model_name)

# 处理样本列表
savefile='sortlabel.txt'
sourcefile='train.json'
f = lambda x:x['label']
llist = TokenList(file=savefile,source=LoadJsons(sourcefile),func=f,low_freq=181,save_low_freq=1)  # 根据12.5的任务4修改为只包含高频标签，如>20
print(f'length of label list {llist.get_num()}')

class MyDataset(Dataset):
    def __init__(self,samples, max_len = 128):
        super().__init__()
        self.data = []
        for tup in samples:
            text = tokenizer([tup[0]],return_tensors='pt',truncation=True,max_length=max_len)['input_ids'][0]
            label = torch.from_numpy(label2vec(tup[1],llist))
            self.data.append( [text, label] )        
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)

# x = ['今天天气真好','明天天气差','好饿']
# y = [[0,1,0],[1,0,1],[1,1,0]]
# dataset = MyDataset(x,y)
def train_model(model, optimizer, train_dl, epochs=3, train_func=None, test_func=None, 
                scheduler=None, save_file=None, accelerator=None, epoch_len=None):  # accelerator：适用于多卡的机器，epoch_len到该epoch提前停止
    best_re = {'acc':0,'prec':0,'reca':0, 'f1':0}
    best_epoch = 0
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
                accu, prec, reca, f1 = test_func()
                if f1 >=best_re['f1'] and save_file:
                    if accelerator:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        accelerator.save(unwrapped_model.state_dict(), save_file)
                    else:
                        torch.save(model.state_dict(), save_file)
                    print(f"Epoch {epoch + 1}, best model saved. Accuracy: {accu:.4f},  Precision: {prec:.4f},  Recall: {reca:.4f},  F1: {f1:.4f}")
                    best_re['f1'] = f1
                    best_re['acc'] = accu
                    best_re['prec'] = prec
                    best_re['reca'] = reca
                    best_epoch = epoch+1
    return best_epoch, best_re
def collate_fn(batch):
    if len(batch[0]) ==2:
        return (nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True),
		    torch.stack([x[1] for x in batch], 0))
    else:
        return (nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True),
		    torch.stack([x[1] for x in batch], 0),torch.cat([x[2] for x in batch])) 
def plot_learning_curve(record):
    y1 = record['train_loss']
    y2 = record['val_f1']
    x1 = np.arange(1,1+len(y1))
    x2 = x1[::int(len(y1)/len(y2))]
    fig = plt.figure(figsize = (6,4))
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
    plt.savefig('100class.png')

def cal_hour(seconds):
    # seconds =35400
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)
epoch = 10
batch_size = 16
lr = 1e-4
ds_train, ds_test = MyDataset(xys[0],max_len= 128), MyDataset(xys[1],max_len=128)
trainloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
testloader = DataLoader(ds_test,batch_size=batch_size,collate_fn=collate_fn)
record = {"train_loss":[],"val_f1":[]}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")
# mfile = 'bs2_largetemp.ckpt'
model = Model_large(n= llist.get_num()).to(device)
# model.load_state_dict(torch.load(mfile,map_location=device))
ckpt = '100class.ckpt'
criterion = nn.BCELoss()
total_steps = len(trainloader) * epoch
optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, lr, total_steps)

def train_func(model, ditem):
    
    xx, yy = ditem[0].to(device), ditem[1].to(device)
    zz = model(xx)
    loss = criterion(zz.float(), yy.float())
    record['train_loss'].append(loss.item())

    return {'loss': loss}
def test_func(): 
    global val_time
    vt1 = time.time()
    yt, yp = [], []
    model.eval()
    with torch.no_grad():
        for xx, yy in testloader:
            #zz = model(xx).detach().cpu().argmax(-1)
            zz = (model(xx.to(device)).detach().cpu()>0.5).float().cpu()
            for z in zz:yp.append(z)
            for y in yy:yt.append(y)
    yp = torch.stack(yp,0).numpy().astype('int64')
    yt = torch.stack(yt,0).numpy().astype('int64')
    accu = accuracy_score(yt,yp)
    #accu = sum([all(yp[i]==yt[i])*1 for i in len(yt)])/len(yt)
    prec = precision_score(yt,yp,average = 'samples')
    reca = recall_score(yt,yp,average = 'samples')
    #f1 = 2*prec*reca/(prec+reca)
    #f1 = 0 if isnan(f1) else f1.item()
    f1 = f1_score(yt,yp,average = 'samples')
    record['val_f1'].append(f1)
    #accu = (np.array(yt) == np.array(yp)).sum() / len(yp)
    stri = f'Accuracy: {accu:.4f},  Precision: {prec:.4f},  Recall: {reca:.4f},  F1: {f1:.4f}'
    print(stri)
    vt2 = time.time()
    val_time += vt2-vt1
    model.train()
    return accu, prec, reca, f1
starttime = time.time()
best_epoch, best_re = train_model(model, optimizer, trainloader, epoch, train_func, test_func, scheduler=scheduler, save_file=ckpt)
plot_learning_curve(record)
endtime = time.time()
print(f'train time:{cal_hour((endtime-starttime-val_time)/epoch)}, val time: {cal_hour(val_time/epoch)}')
print('\n'+f"save best model at epoch {best_epoch}, Accuracy: {best_re['acc']:.4f},  Precision: {best_re['prec']:.4f},  Recall: {best_re['reca']:.4f},  F1: {best_re['f1']:.4f} \n")
print('done')
