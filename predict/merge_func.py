import torch

import torch.nn.functional as F

import model,sortlabel,model_base4,ljqpy

from predict_bs4 import get_embedding

llist = sortlabel.TokenList('datasets/sortlabel.txt', source=ljqpy.LoadJsons('datasets/train.json'), func=lambda x:x['label'], low_freq=1, save_low_freq=1)


# model1 = model.Model()
# model1.load_state_dict(state_dict = torch.load('models/base1cls.ckpt', map_location=torch.device('cpu')))
# model4 = model_base4.Model()
# model4.load_state_dict(state_dict = torch.load('models/cls_base4.pt', map_location=torch.device('cpu')))

class Rules:
    def __init__(self,origin_ret,score,llist):
        self.o_ret = origin_ret
        self.score = score
        self.ts = 0.1
        self.tl = llist
        self.lf = 5
        self.hf = 200 # 频数大于该值的标签，默认200
    def rule1(self):  # 预测的标签都是低频标签，高频标签分数都很低
        ret = False
        
        highfreqlow = all(self.score[:sortlabel.loc(self.hf,self.tl)]<self.ts) 
        
        if sum(self.o_ret[1:]) != 0:
            ret = (self.o_ret[1:].nonzero().min()+1 > sortlabel.loc(self.lf,self.tl)) and highfreqlow
                
        return ret
        


class Val0_Classify():
    def __init__(self,x,attm,tti,y,rep,k,f,model1,model4,mode = 'argmax'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.x = x
        self.attm = attm
        self.tti = tti
        self.y = y
        self.f = f
        if mode == 'argmax':
            self.model = model1
            self.model.to(self.device)
            self.fun = self.argmax_func
        elif mode == 'sentence_pair':
            self.model = model4
            self.model.to(self.device)           
            self.k = k
            self.rep = rep
            self.fun = self.eval_on_val
    
    def argmax_func(self):     
        self.model.eval()
        x = self.x.unsqueeze(0)
        
        with torch.no_grad():               
            x = x.to(self.device)                
            z_index = self.model(x).detach().cpu().argmax(-1).unsqueeze(1)
            z_score = self.model(x).detach().cpu().max()
            z = torch.zeros(1,llist.get_num()).scatter(1,z_index,1).cpu()
            z = z.squeeze(0)
        return z, z_score
    
    def eval_on_val(self,threshold=0.8):        
        self.model.eval()
        input_ids = self.x.unsqueeze(0)
        attention_mask = self.attm.unsqueeze(0)
        token_type_ids = self.tti.unsqueeze(0)
        y = self.y.unsqueeze(0)
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
            z = torch.zeros_like(y).scatter(1,scores,1).cpu()
        return z.squeeze(0)
                     

