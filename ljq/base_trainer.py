import os, sys, time, math, json, re
from tqdm import tqdm
import numpy as np

sys.path.append('../')
import ljqpy, pt_utils

trainfn = 'train_normd.json'
validfn = 'val_normd.json'

import torch
import torch.nn as nn

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'

def LoadData(fn):
    return [(x['text_normd'], x['label']) for x in ljqpy.LoadJsons(fn)]

trains = LoadData(trainfn) 
vals = LoadData(validfn) 
tl = ljqpy.TokenList('sortlabel.txt', low_freq=1)
tl.id2t = tl.id2t[:1] + tl.id2t[2:]
tl.t2id = {v:k for k,v in enumerate(tl.id2t)}

sys.path.append('../plm_training')
tokenizer = pt_utils.GetTokenizer()

class MLP(nn.Module):
    def __init__(self, inp, out) -> None:
        super().__init__()
        self.linear = nn.Linear(inp, out)
    def forward(self, x): return self.linear(x)

from transformers import BertModel
class BertClassification2(nn.Module):
    def __init__(self, n_tags, cls_only=False, plm='hfl/chinese-roberta-wwm-ext') -> None:
        super().__init__()
        self.n_tags = n_tags
        self.encoder = BertModel.from_pretrained(plm)
        self.encoder.load_state_dict(torch.load('pretrain_bert_wb.pt'), strict=False)
        self.pred = MLP(768, n_tags)
    def forward(self, x, seg=None):
        z = self.encoder(x).last_hidden_state[:,0]
        out = self.pred(z)
        #out = torch.sigmoid(out)
        return out

from cetokenizer import CEBertTokenizer, ConvertFlower
import emojiswitch
def Normalize(z): 
    return z.strip()
    z = ConvertFlower(z).strip()
    z = emojiswitch.demojize(z, delimiters=('__',''), lang="zh")
    return z
    return re.sub('__(.+?)__', ' ', z)

class ClsDataset(torch.utils.data.Dataset):
    def __init__(self, samples) -> None:
        super().__init__()
        self.data = []
        for df in samples: self.data.append((Normalize(df[0]), df[1]))
    def __len__(self): return len(self.data)
    def __getitem__(self, k): return self.data[k]

def onehot(xs):
    z = torch.zeros((tl.get_num(),))
    for x in xs: z[x] = 1
    return z

maxlen = 128
def collate_fn(xs):
    tids = tokenizer([x[0] for x in xs], return_tensors='pt', padding=True, truncation=True, max_length=maxlen)['input_ids']
    labels = torch.cat([onehot(map(tl.get_id, x[1])).unsqueeze(0) for x in xs], 0)
    return tids, labels

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], -1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], -1)
    neg_loss = torch.logsumexp(y_pred_neg, -1)
    pos_loss = torch.logsumexp(y_pred_pos, -1)
    return (neg_loss + pos_loss).mean()

def smooth_f1_loss(pred, gold):
    corr = (pred * gold).sum(1)
    return (0.5 * (pred + gold).sum(1) / (corr + 1e-10)).mean()

def m_f1_loss(pred, gold):
    corr = (pred * gold).sum(1)
    return ((pred + gold).sum(1) - corr).mean()

def smooth_f1_loss_linear(pred, gold):
    corr = (pred * gold).sum(1)
    return (1 - (2 * corr + 1) / ((pred + gold).sum(1) + 1)).mean()

def sample_f1_loss(pred, gold):
    corr = (pred * gold).sum(1)
    pp = corr / pred.sum(1, keepdim=True)
    rr = corr / gold.sum(1, keepdim=True)
    return ((pp + rr) / (pp * rr * 2 + 1e-10) - 1).mean()

def sample_f1_loss_linear(pred, gold):
    corr = (pred * gold).sum(1)
    pp = corr / pred.sum(1, keepdim=True)
    rr = corr / gold.sum(1, keepdim=True)
    return (1 - (2 * pp * rr) / (pp + rr + 1e-10)).mean()

def list_f1(s1, s2):
    s1, s2 = set(s1), set(s2)
    ss = s1 & s2
    if len(ss) == 0: return 0
    p, r = len(ss) / len(s1), len(ss) / len(s2)
    return round(2 * p * r / (p + r), 4)

if __name__ == '__main__':
    ds_train = ClsDataset(trains)
    ds_test = ClsDataset(vals)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=2)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=64, collate_fn=collate_fn)

    model = BertClassification2(n_tags=tl.get_num(), cls_only=True)
    
    pt_utils.lock_transformer_layers(model.encoder, 10)

    mfile = 'wb_base2_retrain2_lock10_diceloss_normd1.pt'
    #if os.path.exists(mfile): model.load_state_dict(torch.load(mfile))

    epochs = 35
    total_steps = len(dl_train) * epochs

    import accelerate
    from accelerate import Accelerator, DistributedDataParallelKwargs
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    #optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, 5e-5, total_steps)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    optimizer = accelerate.utils.DummyOptim(optimizer_grouped_parameters)
    scheduler = accelerate.utils.DummyScheduler(optimizer, total_num_steps=total_steps, warmup_num_steps=total_steps//10)

    model, optimizer, dl_train, scheduler = accelerator.prepare(model, optimizer, dl_train, scheduler)

    device = accelerator.device
    model.to(device)

    bce_loss_func = nn.BCELoss()
    loss_func = lambda y, z: -(z*torch.log(y+1e-12)*2 + (1-z)*torch.log(1-y+1e-12)).mean()
    pu_loss_fct = lambda y_pred, y_true: - 10*(y_true*torch.log(y_pred+1e-9)).mean() - torch.log(((1-y_true)*(1-y_pred)).mean()+1e-9)
    mcc_loss_func = multilabel_categorical_crossentropy

    relu = nn.ReLU()
    def train_func(model, ditem):
        x, y = ditem
        y = y.to(device)
        out = model(x.to(device))
        #loss = 10 * loss_func(out, y) + 5 * smooth_f1_loss_linear(out, y) # + relu(out.mean() - 0.1)
        loss = mcc_loss_func(out, y) 
        out = torch.sigmoid(out)
        loss += smooth_f1_loss_linear(out, y)
        oc = (out.detach() > 0.5).float()
        prec = (oc + y > 1.5).sum() / max(oc.sum().item(), 1)
        reca = (oc + y > 1.5).sum() / max(y.sum().item(), 1)
        f1 = 2 * prec * reca / (prec + reca)
        r = {'loss': loss, 'prec': prec, 'reca': reca, 'f1':f1}
        return r

    def test_func(): 
        outs = [];  ys = []; outv = []
        with torch.no_grad():
            for x, y in dl_test:
                out = model(x.to(device)) 
                out = torch.sigmoid(out)
                outv.append(out.detach().clone().cpu())
                out = (out > 0.5).long().detach().cpu()
                outs.append(out)
                ys.append(y)
        outv = torch.cat(outv, 0)
        outs = torch.cat(outs, 0)
        ys = torch.cat(ys, 0).cpu()
        accu = (outs == ys).float().mean()
        prec = (outs + ys == 2).float().sum() / outs.sum()
        reca = (outs + ys == 2).float().sum() / ys.sum()
        f1 = 2 * prec * reca / (prec + reca)
        if accelerator is None or accelerator.is_local_main_process:
            print(f'Accu: {accu:.4f},  Prec: {prec:.4f},  Reca: {reca:.4f},  F1: {f1:.4f}')
            data = []; corr = 0
            for x, out in zip(ljqpy.LoadJsons(validfn), outv):
                pp = []
                for index in (out > 0.1).nonzero():
                    pp.append(tl.get_token(index))
                if len(pp) == 0: 
                    mx = out.argmax()
                    vv = out[mx].item()
                    thre = vv * 0.5
                    for index in (out > thre).nonzero():
                        pp.append(tl.get_token(index))
                    #pp.append(tl.get_token(mx))
                    x['back'] = round(vv, 4)
                x['pred'] = pp
                x['id'] = list_f1(pp, x['label'])
                if ''.join(sorted(pp)) == ''.join(sorted(list(set(x['label'])))): corr += 1
                #else: x['id'] = -1
                x.pop('text_normd')
                data.append(x)
            print(f'corr: {corr}, accu: {corr/len(data):.4f}')
            data.sort(key=lambda x:x['id'])
            ljqpy.SaveJsons(data, 'output.json')
            os.system('python evaluate.py')

    pt_utils.train_model(model, optimizer, dl_train, epochs, train_func, test_func, 
                scheduler=scheduler, save_file=mfile, accelerator=accelerator)
    #model.eval(); test_func()

    print('done')