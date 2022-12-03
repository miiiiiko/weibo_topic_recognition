import os, sys, time, math
from tqdm import tqdm
import numpy as np

dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(dir, '../utils'))
import ljqpy, pt_utils

tokenizer = pt_utils.GetTokenizer()
import torch
import torch.nn as nn

class Sentiment:
	def __init__(self, tl=None, device='cpu', mfile='sentiment.pt'):
		self.tl = tl
		if tl is None: self.tl = ljqpy.TokenList(os.path.join(dir, 'labellist.txt'), special_marks=None) 
		self.model = pt_utils.BERTClassification(n_tags=self.tl.get_num(), cls_only=True)
		self.device = device
		self.model.to(device)
		if tl is None: 
			self.model.load_state_dict(torch.load(os.path.join(dir, mfile), map_location=device))
			self.model.eval()
		
	def predict(self, sents):
		xs = tokenizer(sents, return_tensors='pt', padding=True).input_ids
		with torch.no_grad():
			zz = self.model(xs.to(self.device)).detach().cpu().argmax(-1).numpy()
		ret = []
		for s, z in zip(sents, zz):
			ret.append(self.tl.get_token(z))
		return ret

if __name__ == '__main__':
	ss = Sentiment(device='cuda')
	print(ss.predict(['其实我愿意吞声忍气，但却怕被看不起。', '虽然肚子疼，可还是要谢谢你，爱你！']))