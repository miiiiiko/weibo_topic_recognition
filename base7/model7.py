import torch
import torch.nn as nn
from transformers import BertModel,ErnieForMaskedLM

class MLP(nn.Module):
    def __init__(self, n_in, n_out): # dropout=0
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        # self.activation = nn.Sigmoid()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = self.dropout(x)
        x = self.linear(x)
        # x = self.activation(x)
        return x


class Model7(nn.Module):
    def __init__(self, model_path='hfl/chinese-roberta-wwm-ext',n=1400,encoder_type = 'cls', ernie=False):
        super(Model7, self).__init__()
        self.encoder_type = encoder_type
        if ernie:
            self.encoder = ErnieForMaskedLM.from_pretrained("nghuyong/ernie-3.0-base-zh")
        else:
            self.encoder = BertModel.from_pretrained(model_path)
        self.pred = MLP(768,n)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def forward(self,inputs):
        x = self.encoder(inputs)
        if self.encoder_type == "cls":
            x = x.last_hidden_state[:,0]
        if self.encoder_type == "pooler":
            x = x.pooler_output
        x = self.pred(x)
        return x

class Model_large(nn.Module):
    def __init__(self, model_path='hfl/chinese-roberta-wwm-ext-large',n=1400,encoder_type = 'cls',ernie=False):
        super(Model_large, self).__init__()
        self.encoder_type = encoder_type
        if ernie:
            self.encoder = ErnieForMaskedLM.from_pretrained("nghuyong/ernie-3.0-xbase-zh")
        else:
            self.encoder = BertModel.from_pretrained(model_path)
        # self.encoder = BertModel.from_pretrained(model_path)
        self.pred = MLP(1024,n)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    def forward(self,inputs):
        x = self.encoder(inputs)
        if self.encoder_type == "cls":
            x = x.last_hidden_state[:,0]
        if self.encoder_type == "pooler":
            x = x.pooler_output
        x = self.pred(x)
        return x

