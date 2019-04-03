from model import resnet18
import torch
from data import preprocess_strategy,ValFolder
import json
import os
import tqdm
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]='0'
trans=preprocess_strategy(dataset='tsne')
valset=ValFolder(root='../datafolder/c9/all/',transform=trans,with_path=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=50, shuffle=False, num_workers=32,pin_memory=True)
model=resnet18(pretrained=True,num_classes=9)
model.cuda()

data={}
data['path']=[]
data['feature']=[]
data['label']=[]
model.eval()
with torch.no_grad():
	for input,target,path in tqdm.tqdm(valloader):
		input=input.cuda()
		_,_,out=model(input)
		data['path'].extend(path)
		data['feature'].extend(out.cpu().numpy().tolist())
		data['label'].extend([int(i) for i in target.numpy()])

with open('data.json','w') as out:
	json.dump(data,out)