import os, sys, time, math, ljqpy, random
from tqdm import tqdm
import numpy as np
dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append('/data1/ljq/')
sys.path.append('/home/jqliang/')
import pt_utils

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'

import torch
import torch.nn as nn

sys.path.append('../plm_training')

from cetokenizer import CEBertTokenizer, ConvertFlower
#tokenizer = CEBertTokenizer('../plm_training/vocab.txt')
#from GAU import MLMHeadModel, GAUNet
#model = MLMHeadModel(GAUNet(vocab_size=tokenizer.vocab_size))

from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertModel
tokenizer = pt_utils.GetTokenizer()
model = BertForMaskedLM.from_pretrained('hfl/chinese-roberta-wwm-ext')

maxlen = 128

def collate_fn(xs):
    return {k:torch.tensor([x[k] for x in xs], dtype=torch.long) for k in xs[0].keys()}

def WeiboSentences(maxlen=510, minlen=4):
    trainfn = 'train_normd.json'
    valfn = 'val_normd.json'
    testfn = 'test_normd.json'
    tsents = [x['text_normd'] for x in ljqpy.LoadJsons(trainfn)+ljqpy.LoadJsons(valfn)+ljqpy.LoadJsons(testfn)]
    while True:
        random.shuffle(tsents)
        for x in tsents:
            sent = Normalize(x.strip())
            for i in range(0, len(sent), maxlen):
                z = sent[i:i+maxlen]
                if len(z) > minlen: yield z

from corpus_dataset import PureGenDataset, RoBERTaFullSentFast
import emojiswitch, re
#def Normalize(z):
#    z = ConvertFlower(z).strip()
#    return emojiswitch.demojize(z, delimiters=('#',''), lang="zh")

def Normalize(z):
    return z.strip()
    z = ConvertFlower(z).strip()
    z = emojiswitch.demojize(z, delimiters=('__','__'), lang="zh")
    return re.sub('__(.+?)__', ' ', z)

if __name__ == '__main__':
    bsz = 32
    #ds = GeneratorWWMDataset([CLUESenteces, WikiSenteces], bsz*20000, maxlen, tokenizer)
    ds = PureGenDataset(RoBERTaFullSentFast(WeiboSentences(maxlen-2, 1), tokenizer, maxlen, repeat=1), bsz*3000)
    dl = torch.utils.data.DataLoader(ds, batch_size=bsz, collate_fn=collate_fn, num_workers=2)

    mfile = 'pretrain_bert_mlm_wb.pt'
    #model.load_state_dict(torch.load('../plm_training/gau_mlm_full.pt'))
    model.load_state_dict(torch.load(mfile))
    #torch.save(model.bert.state_dict(), mfile.replace('_mlm', ''))
    #sys.exit()
    epochs = 2
    total_steps = len(dl) * epochs

    import accelerate
    from accelerate import Accelerator, DistributedDataParallelKwargs
    accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)])

    #optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, 1e-4, total_steps)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
    ]
    optimizer = accelerate.utils.DummyOptim(optimizer_grouped_parameters)
    scheduler = accelerate.utils.DummyScheduler(optimizer, total_num_steps=total_steps, warmup_num_steps=total_steps//10)
    model, optimizer, dl_train, scheduler = accelerator.prepare(model, optimizer, dl, scheduler)

    device = accelerator.device
    model.to(device)

    def train_func(model, x):
        attent_mask = (x['input_ids'] > 0).long()
        out = model(input_ids=x['input_ids'].to(device), 
                token_type_ids=x['token_type_ids'].to(device),
                attention_mask=attent_mask.to(device),
                labels=x['labels'].to(device),
                return_dict=True)
        return out.loss

    def test_func(): pass

    pt_utils.train_model(model, optimizer, dl, epochs, train_func, test_func, 
                scheduler=scheduler, save_file=mfile, accelerator=accelerator)

    torch.save(model.bert.state_dict(), mfile.replace('_mlm', ''))
    print('done')