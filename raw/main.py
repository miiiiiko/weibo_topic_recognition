import torch
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
#from easonsi import utils
from ljqpy import LoadJsons
from ljqpy import TokenList
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import Model
import torch.nn as nn
from torch.optim import AdamW
import transformers
from tqdm import tqdm
from sort_label import label2vec
from sklearn import metrics
import time


def load_data(d_path):
    data = LoadJsons(d_path)
    text = []
    labels = []
    for d in data:
        text.append(d["text_normd"])
        labels.append(label2vec(d["label"]))
    return text,labels



class MyDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __getitem__(self, index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return len(self.x)


# x = ['今天天气真好','明天天气差','好饿']
# y = [[0,1,0],[1,0,1],[1,1,0]]
# dataset = MyDataset(x,y)


model_name = 'hfl/chinese-roberta-wwm-ext'



def collate_batch(batch):
    tokenizer = BertTokenizer.from_pretrained(model_name) 
    x,y = zip(*batch)
    
    sen_code = tokenizer(x, padding=True, return_tensors ='pt')
    input_ids = sen_code['input_ids']
    token_type_ids = sen_code['token_type_ids']
    attention_mask = sen_code['attention_mask']
    return input_ids,token_type_ids,attention_mask, torch.FloatTensor(y) # .long()
  

def get_dataloader(dataset,batch_size,n_workers=8):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=n_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    return dataloader
  
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataloader = DataLoader(dataset,batch_size=2,shuffle=False,collate_fn=collate_batch)
# model = Model().to(device)
# for batch in dataloader:
#     input_ids,token_type_ids,attention_mask,y = batch
#     input_ids,attention_mask,token_type_ids = input_ids.to(device),attention_mask.to(device),token_type_ids.to(device)
#     y = model(input_ids,attention_mask,token_type_ids)
#     print(y)

#     break
train_set = load_data('/home/qsm22/datasets/weibo_topic_recognition/train_normd.json')
train_loader = get_dataloader(train_set,batch_size=32,n_workers=8)

def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""

    input_ids,token_type_ids,attention_mask,labels = batch
    input_ids,attention_mask,token_type_ids = input_ids.to(device),attention_mask.to(device),token_type_ids.to(device)
    labels = labels.to(device)

    outs = model(input_ids,attention_mask,token_type_ids)

    loss = criterion(outs, labels)
    f1 = eval_on_batch(batch,model,device)
    return loss,f1

def cal_hour(seconds):
    # seconds =35400
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("%d:%02d:%02d" % (h, m, s))
    return

def plot_learning_curve(record):
    config = parse_args()
    x1 = np.arange(config['total_steps'])
    x2 = x1[::config['valid_steps']]
    y1 = record['train_loss']
    y2 = record['val_f1']
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
    plt.savefig('learning_curve')

def parse_args():
    """arguments"""
    config = {
        "data_dir": "../input/ml2021springhw43/Dataset/",
        "model_name": 'hfl/chinese-roberta-wwm-ext',
        "save_path": "model.ckpt",
        "batch_size": 64,
        "n_workers": 8,
        "valid_steps": 20,
        "warm_factor": 0.05,
        "save_steps": 100,
        "total_steps": 100,
    }

    return config

def eval_on_batch(batch,model,device):
    model = model.to(device)
    input_ids,token_type_ids,attention_mask, test_y = batch
    input_ids,attention_mask,token_type_ids = input_ids.to(device),attention_mask.to(device),token_type_ids.to(device)
    test_output = model(input_ids,attention_mask,token_type_ids).cpu()
    # test_y = test_y.to(device)
    zero = torch.zeros_like(test_output) 
    one = torch.ones_like(test_output)   
    test_output = torch.where(test_output > 0.5, one, test_output)
    pred = torch.where(test_output < 0.5, zero, test_output)
    y_true = test_y.detach().numpy().astype("int64")
    y_pred = pred.detach().numpy().astype("int64")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    f1_score = metrics.f1_score(y_true, y_pred, average='samples')
    return f1_score


def val(test_loader,model,device):
    # net = model.to(device)
    model = model.to(device)
    f1_l = []
    # y_true,y_pred = [],[]
    for stp, (input_ids,token_type_ids,attention_mask, test_y) in enumerate(test_loader):
        # y_true,y_pred = [],[]
        input_ids,attention_mask,token_type_ids = input_ids.to(device),attention_mask.to(device),token_type_ids.to(device)
        test_output = model(input_ids,attention_mask,token_type_ids).cpu()
        # test_y = test_y.to(device)
        zero = torch.zeros_like(test_output) 
        one = torch.ones_like(test_output)   
        test_output = torch.where(test_output > 0.5, one, test_output)
        pred = torch.where(test_output < 0.5, zero, test_output)
        # # pred = [int(x>0.5) for x in test_output]
        # y_pred = []
        # for d in test_output:
        #     pred = [int(x>0.5) for x in d]
        #     y_pred.append(pred)

        

        y_true = test_y.detach().numpy().astype("int64")
        y_pred = pred.detach().numpy().astype("int64")

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    # print(y_true)
    # print(y_pred)
    # print(y_true.dtype,y_pred.dtype)
        # y_pred = y_pred.cpu()
        # precision = metrics.precision_score(y_true, y_pred, average='samples')
        # recall = metrics.recall_score(y_true, y_pred, average='samples')
        f1_score = metrics.f1_score(y_true, y_pred, average='samples')
        f1_l.append(f1_score)
    return sum(f1_l)/len(f1_l)


def main(
  data_dir,
  model_name,
  save_path,
  batch_size,
  n_workers,
  valid_steps,
  warm_factor,
  total_steps,
  save_steps,

):
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    record = {'train_loss':[],'val_f1':[]}
    print(f"[Info]: Use {device} now!")
    train_data = load_data('/home/qsm22/datasets/weibo_topic_recognition/train_normd.json')
    val_data = load_data('/home/qsm22/datasets/weibo_topic_recognition/val_normd.json')
    # train_data = load_data('test.json')
    # val_data = load_data('test.json')
    train_set = MyDataset(train_data[0],train_data[1])
    val_set = MyDataset(val_data[0],val_data[1])
    train_loader = get_dataloader(train_set,batch_size=batch_size,n_workers=n_workers)
    valid_loader = get_dataloader(val_set,batch_size=batch_size,n_workers=n_workers)
    train_iterator = iter(train_loader)                               #这里没有懂为什么要用iter(), train_loader本身就是一个iterator吧
    print(f"[Info]: Finish loading data!",flush = True)

    model = Model(model_name, 1400).to(device)
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=1e-4,weight_decay=1e-5)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=warm_factor * total_steps,
                                                            num_training_steps=total_steps)

    print(f"[Info]: Finish creating model!",flush = True)                     # flush = True是什么意思

    best_f1 = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")
    time1 = time.time()
    val_time = 0
    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss,f1 = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_f1 = f1.item()
        record['train_loss'].append(batch_loss)

        # Updata model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Log
        pbar.update()
        pbar.set_postfix(
        loss=f"{batch_loss:.2f}",
        f1=f"{batch_f1:.2f}",
        step=step + 1,
        )

        # Do validation
        val_time1 = time.time()
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_f1 = val(valid_loader, model, device)
            record['val_f1'].append(valid_f1)

            # keep the best model
            if valid_f1 > best_f1:
                best_f1 = valid_f1
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (f1={best_f1:.4f})")
        val_time2 = time.time()
        val_time += val_time2 - val_time1

    pbar.close()
    time2 = time.time()
    print('训练时间')
    cal_hour(time2-time1-val_time)
    print('验证时间')
    cal_hour(val_time)
    plot_learning_curve(record)
    # print(time2-time1-val_time, val_time)



if __name__ == "__main__":

  main(**parse_args())

    