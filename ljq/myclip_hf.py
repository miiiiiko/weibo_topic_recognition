import ljqpy, re, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import base64
from tqdm import tqdm
from torchvision import transforms
import numpy as np

from transformers import BertTokenizer, BertModel, CLIPModel 
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPProcessor
tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')

from transformers import logging
logging.set_verbosity_error()

version = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(version)

transform_train = transforms.Compose([
    transforms.Resize(224, transforms.InterpolationMode.BICUBIC),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def flatNCE_loss(logits):
    b = logits.shape[0]
    offdiag = logits.flatten()[:-1].view(b-1, b+1)[:,1:].reshape(b, b-1)
    diag = torch.diag(logits)
    floss = torch.logsumexp(offdiag, dim=-1) - diag
    return floss.mean()

class MyCLIP(nn.Module):
    def __init__(self, version="openai/clip-vit-large-patch14") -> None:
        super().__init__()
        self.clip = CLIPModel.from_pretrained(version)
        self.text_encoder = BertModel.from_pretrained('hfl/chinese-roberta-wwm-ext')

    def encode_text(self, input_ids):
        seg = torch.zeros_like(input_ids)
        return self.text_encoder(input_ids, token_type_ids=seg).last_hidden_state

    def forward(self, input_ids, pixel_values):
        with torch.no_grad():
            image_embeds = self.clip.get_image_features(pixel_values)
        text_embeds = self.encode_text(input_ids)
        text_embeds = self.clip.text_projection(text_embeds[:,0])
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        loss_fn = flatNCE_loss
        caption_loss = loss_fn(logits_per_text)
        image_loss = loss_fn(logits_per_image)
        loss = (caption_loss + image_loss) / 2.0
        return loss

if __name__ == '__main__':
    mfile = 'myclip_hf.ckpt'
    clip = MyCLIP()
    clip.load_state_dict(torch.load(mfile))
    torch.save(clip.text_encoder.state_dict(), 'myclip_textenc.ckpt')
    sys.exit()
    ctokenizer = CLIPTokenizer.from_pretrained(version)
    clip = CLIPModel.from_pretrained(version).cuda()
    ctext = CLIPTextModel.from_pretrained(version).cuda()
    cvision = CLIPVisionModel.from_pretrained(version).cuda()

    text = 'A cute happy girl with long hairs and beautiful flowers'
    batch_encoding = ctokenizer(text, truncation=True, max_length=77, return_length=True,
                                return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"].cuda()

    out1 = ctext(tokens).last_hidden_state
    print(out1.shape)
    out2 = clip.get_text_features(tokens)
    print(out2.shape)

    import requests
    from PIL import Image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    zz = transform_train(image)
    print(zz)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    print(inputs)


    #DetectPretrained()
    clip.load_state_dict(torch.load(mfile))
    clip = clip.cuda().eval()
    def SimpleTest():
        for ditem in dl_train: break
        image, text = ditem[0].cuda(), ditem[1].cuda()
        with torch.no_grad():
            image_features = clip.encode_image(image)
            text_features = clip.encode_text(text)
            logits_per_image, logits_per_text = clip(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            print(probs[0])


    allvecs = torch.zeros( len(ds), 512 )
    with torch.no_grad():
        ii = 0
        for img, text in tqdm(dl_search):
            img_feas = clip.encode_image(img.cuda()).detach().cpu()
            allvecs[ii:ii+img_feas.shape[0]] = img_feas
            ii += img_feas.shape[0]

    print(allvecs.shape)     
    allvecs /= allvecs.norm(dim=-1, keepdim=True)   
    text = '非常下饭非常鲜美的大蒜豆豉烧黄鱼，每一口都会让你齿颊留香'
    while True:
        tt = tokenizer.encode(text, max_length=77, padding='max_length', truncation=True)
        tt = torch.tensor([tt]).cuda()
        with torch.no_grad():
            text_feas = clip.encode_text(tt).detach().cpu()
        print(text_feas.shape)
        text_feas /= text_feas.norm(dim=-1, keepdim=True)
        similarity = (100.0 * text_feas @ allvecs.T).softmax(dim=-1)
        values, indices = similarity[0].topk(5)
        for i, (value, index) in enumerate(zip(values, indices)):
            value, index = value.item(), index.item()
            print(i, index, value, ds.items[index])
            fn = f'{i}_{value:.4f}.png'
            saved = ds.saved[ds.items[index]]
            if type(saved) is type(''):
                pimg = Base64ToImg(saved)
            else: pimg = saved[1]
            pimg.save(fn)
        text = input('> ')
    
