import torch
import torch.nn as nn
from transformers import BertTokenizer,BertModel

class MLP(nn.Module):
    def __init__(self, n_in, n_out): # dropout=0
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.Sigmoid()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class Model(nn.Module):
    def __init__(self, model_path='hfl/chinese-roberta-wwm-ext',n=1400):
        super(Model, self).__init__()
        self.encoder = BertModel.from_pretrained(model_path)
        self.pred = MLP(768,n)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def forward(self,input_ids,attention_mask,token_type_ids):
        x = self.encoder(input_ids,attention_mask,token_type_ids).pooler_output
        x = self.pred(x)
        return x


