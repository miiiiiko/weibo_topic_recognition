import os, sys, time, math, json
from tqdm import tqdm
import numpy as np

dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(dir, '../utils'))
import ljqpy, pt_utils

from sentiment import Sentiment, tokenizer

#datadir = os.path.join(dir, '../../traindata/CLUEmotionAnalysis2020-master')
datadir = os.path.join(dir, '../../traindata/FinFE')

import torch
import torch.nn as nn

def LoadData(fn):
	return [(x['content'], x['label']) for x in ljqpy.LoadJsons(fn)]

def LoadData(fn):
	with open(fn, 'r', encoding='utf-8') as fin:
		d = json.loads(fin.read())
	vl = '负面,中立,正面'.split(',')
	return [(x[0], vl[x[1]]) for x in d]

xys = [LoadData(os.path.join(datadir, '%s_list.json') % tp) for tp in ['train', 'test']]
tl = ljqpy.TokenList('labellist.txt', source=xys[0], func=lambda x:[x[1]], special_marks=None)

class ClsDataset(torch.utils.data.Dataset):
	def __init__(self, samples, tl, maxlen=128) -> None:
		super().__init__()
		self.data = []
		for df in samples:
			text = tokenizer([df[0]], return_tensors='pt', truncation=True, max_length=maxlen)['input_ids'][0]
			label = torch.tensor([tl.get_id(df[1])])
			self.data.append( [text, label] )
	def __len__(self): return len(self.data)
	def __getitem__(self, k): return self.data[k]

def collate_fn(xs):
    return (nn.utils.rnn.pad_sequence([x[0] for x in xs], batch_first=True),
			torch.cat([x[1] for x in xs], 0))


if __name__ == '__main__':
	ds_train, ds_test = ClsDataset(xys[0], tl), ClsDataset(xys[1], tl)
	dl_train = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True, collate_fn=collate_fn)
	dl_test = torch.utils.data.DataLoader(ds_test, batch_size=64, collate_fn=collate_fn)

	sentiment = Sentiment(tl)
	sentiment.model.cuda()
	model = sentiment.model

	mfile = 'sentiment.pt'

	epochs = 3
	total_steps = len(dl_train) * epochs

	optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, 1e-4, total_steps)
	loss_func = nn.CrossEntropyLoss()
	
	def train_func(model, ditem):
		xx, yy = ditem[0].cuda(), ditem[1].cuda()
		zz = model(xx)
		loss = loss_func(zz, yy)
		return {'loss': loss}

	def test_func(): 
		yt, yp = [], []
		model.eval()
		with torch.no_grad():
			for xx, yy in dl_test:
				xx, yy = xx.cuda(), yy
				zz = model(xx).detach().cpu().argmax(-1)
				for y in yy: yt.append(y.item())
				for z in zz: yp.append(z.item())
		accu = (np.array(yt) == np.array(yp)).sum() / len(yp)
		print(f'Accu: {accu:.4f}')
		model.train()

	pt_utils.train_model(model, optimizer, dl_train, epochs, train_func, test_func, 
				scheduler=scheduler, save_file=mfile)

	print('done')