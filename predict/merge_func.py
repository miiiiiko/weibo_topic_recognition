import sys
sys.path.append('.')
# sys.path.append('./base4')
import torch

import torch.nn.functional as F

from base1 import sortlabel,ljqpy

# from base4.predict import get_embedding

llist = sortlabel.TokenList('dataset/sortlabel.txt', source=ljqpy.LoadJsons('dataset/train.json'), func=lambda x:x['label'], low_freq=1, save_low_freq=1)


# model1 = model.Model()
# model1.load_state_dict(state_dict = torch.load('models/base1cls.ckpt', map_location=torch.device('cpu')))
# model4 = model_base4.Model()
# model4.load_state_dict(state_dict = torch.load('models/cls_base4.pt', map_location=torch.device('cpu')))

class Rules:
    def __init__(self,origin_ret,score,llist):
        self.o_ret = origin_ret     # original result
        self.score = score # original score
        self.ts = 0.1 # threshold
        self.tl = llist
        self.lf = 5 # low_freq
        self.hf = 200 # high_freq 频数大于该值的标签即为高频标签，默认200
    def rule1(self):  # 预测的标签都是低频标签，高频标签分数都很低
        ret = False
        
        highfreqlow = all(self.score[:sortlabel.loc(self.hf,self.tl)]<self.ts) 
        
        if sum(self.o_ret[1:]) != 0:
            ret = (self.o_ret[1:].nonzero().squeeze(1).min() >= sortlabel.loc(self.lf,self.tl)) and highfreqlow
                
        return ret #True or False
        


class Val0_Classify():
    def __init__(self,llist, rep=None,k=3,f=5,model4=None,mode = 'argmax'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tl = llist
        
        if mode == 'argmax': # 因为model1已经加载好并计算出成绩不需要重复加载
            self.fun = self.argmax_func

        elif mode == 'sentence_pair': # 此时需要加载模型model4
            self.f = f
            self.k = k
            self.model = model4
            self.model.to(self.device)           
            self.rep = rep
            self.fun = self.sentence_pair_func
    
    def argmax_func(self,score_i):
        # 此方法每次输入一个样本的成绩，将预测的标签与分数打包在一起。
        # score, idx = torch.sort(score_i,descending=True)
        # if score[1] > 0.4*score[0]:
        #     pred_idx = idx[:2]
        # else:
        #     pred_idx = idx[0].unsqueeze(0)
        # pred_vec = torch.zeros(self.tl.get_num()).scatter(0,pred_idx,1)
        # return pred_vec  
        pred_vec = torch.zeros_like(score_i)
        threshold = score_i.max()*0.4
        for idx in (score_i > threshold).float().nonzero().squeeze(1):
            pred_vec[idx] = 1
        return pred_vec

        

    def sentence_pair_func(self,x, attm, tti, threshold=0.8):        
        self.model.eval()
        input_ids = x.unsqueeze(0)
        attention_mask = attm.unsqueeze(0)
        token_type_ids = tti.unsqueeze(0)
        
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            scores = torch.zeros(1,llist.get_num())  # 1*1400
            
            yy = F.normalize(self.model(input_ids,attention_mask,token_type_ids)).cpu()  # 1*768
            # print(yy.shape)
            # print(self.rep.T.shape)
            cosine_scores = torch.mm(yy,self.rep.T)  # rep.T: 768*(1399*3),cos_score: 1*4197
            for i in range(len(cosine_scores[0])):  # range(4197)
                j = sortlabel.loc(self.f,llist) + i//3  # loc(max,llist) = 1
                scores[:,j] += cosine_scores[:,i]
            # print(scores[:,0])
            # scores = (scores/self.k > threshold).float()  # 取阈值
            scores = (scores/self.k).argmax(-1).unsqueeze(1)  # 取最大值
            z = torch.zeros(1,llist.get_num()).scatter(1,scores,1).cpu()
        return z.squeeze(0)