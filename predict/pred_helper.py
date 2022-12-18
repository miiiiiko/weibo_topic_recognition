import sys
sys.path.append('.')
import torch
import ljqpy
from main import llist


all_data = ljqpy.LoadJsons('./dataset/train.json')
label_matrix = torch.zeros(1400,1400)
for d in all_data:
    label_list = d['label']
    for i in range(len(label_list)):
        for j in range(len(label_list)):
            l1 = llist.get_id(label_list[i])
            l2 = llist.get_id(label_list[j])
            label_matrix[l1][l2] += 1

label_matrix = torch.stack([ele/ele.sum() for ele in label_matrix])
label_pair = (label_matrix == 1).nonzero()
# print(len(label_pair))
# print(label_pair)
def transfer(scores):
    for i in range(len(label_pair)):
        i,j = label_pair[i][0],label_pair[i][1]
        # print(i,j)
        res = torch.max(scores[...,i],scores[...,j])
        scores[...,i],scores[...,j] = res,res
    return scores
# scores = torch.tensor([[0]*1400,[1]*1400])

# transfer(scores)
def transfer_z(z):
    for i,j in label_pair:
        if z[i] == 1 or z[j] == 1:
            z[i],z[j] = 1,1
    return z
    # if zz[idx][78]==1 or zz[idx][79]==1 or scores[idx][78] + scores[idx][79] > 0.5:
                    #     zz[idx][78],zz[idx][79] = 1,1 

# 'label_8262':'绝对舞台掌控者'，'label_805278':'常西西'
lab2word = {'label_895459':'入境','label_582805':'情侣头像','label_493566':'萧邦',
            'label_766890':'单身久了', 'label_819030':'张俪','label_1219604':'堂食','label_1340977':'民族匠心','label_361083':'反诈老陈',
            'label_658167':'迷你世界','label_137439':'恋恋北极星','label_1390048':'航空公司','label_207137':'李紫婷','label_161121':'金译团',
            'label_1003895':'光子','label_161726': '爱情电影频道','label_102402': '六一儿童节',"label_1064693":"韩国炸鸡",'label_312604':'华为深圳','label_596430':'榴莲',
            'label_1516519':'姜栋元','label_247451':'天猫618','label_1313425':'天猫618','label_206611':'小麦','label_1429838':'夏娃','label_714824':'金湾'}

def check_text(text,lab2word=lab2word):
    indices = []
    for k,v in lab2word.items():
        if v in text:
            indices.append(llist.get_id(k))
    return indices
