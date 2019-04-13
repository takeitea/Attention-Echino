from torch import nn
from .resnet import resnet50
import torch
import torch.nn.functional as F
import numpy as np
from .anchor import generator_default_anchor_maps,hard_nms
from .BCNN import BCNN
import pretrainedmodels
CAT_NUM=4
PROPOSAL_NUM=6

class ProposalNet(nn.Module):
	def __init__(self):
		super(ProposalNet,self).__init__()
		self.down1=nn.Conv2d(512*4,128,3,1,1)
		self.down2=nn.Conv2d(128,128,3,2,1)
		self.down3=nn.Conv2d(128,128,3,2,1)

		self.relu=nn.ReLU()
		self.tidy1=nn.Conv2d(128,6,1,1,0)
		self.tidy2=nn.Conv2d(128,6,1,1,0)
		self.tidy3=nn.Conv2d(128,9,1,1,0)

	def forward(self, x):
		batch_size=x.size(0)
		d1=self.relu(self.down1(x))
		d2=self.relu(self.down2(d1))
		d3=self.relu(self.down3(d2))
		t1=self.tidy1(d1).view(batch_size,-1)
		t2=self.tidy2(d2).view(batch_size,-1)
		t3=self.tidy3(d3).view(batch_size,-1)
		return torch.cat((t1,t2,t3),dim=1)

class Attention_Net(nn.Module):
	def __init__(self,topN=4,num_classes=9):
		super(Attention_Net,self).__init__()
		self.pretrained_model=resnet50(pretrained=True)
		self.pretrained_model.avgpool=nn.AdaptiveAvgPool2d(1)
		self.pretrained_model.fc=nn.Linear(512*4,num_classes)
		self.proposal_net=ProposalNet()
		self.topN=topN
		self.concat_net=nn.Linear(512*(CAT_NUM+1)*4,num_classes)
		self.partcls_net=nn.Linear(512*4,num_classes)
		_,edge_anchors,_=generator_default_anchor_maps()
		# TODO  next line
		self.pad_size=224
		self.edge_anchors=(edge_anchors+224).astype(np.int)
		# self.lstm1=nn.LSTM(512,128,batch_first=True,num_layers=1,bidirectional=False,bias=False)
		# self.lstm2=nn.LSTM(128,128,batch_first=True,num_layers=1,bidirectional=False,bias=False)

	def forward(self, x):
		resnet_out,rpn_feature,feature=self.pretrained_model(x)
		x_pad=F.pad(x,(self.pad_size,self.pad_size,self.pad_size,self.pad_size),mode='constant',value=0)

		batch=x.size(0)
		rpn_score=self.proposal_net(rpn_feature.detach())
		all_cdds=[ np.concatenate((x.reshape(-1,1),self.edge_anchors.copy(),np.arange(0,len(x)).reshape(-1,1)),axis=1)
								  for x in rpn_score.data.cpu().numpy()]
		# TODO the threshold
		top_n_cdds=[hard_nms(x,topn=self.topN,iou_threshold=0.25) for x in all_cdds]
		top_n_cdds=np.array(top_n_cdds)
		top_n_index=top_n_cdds[:,:,-1].astype(np.int)
		top_n_index=torch.from_numpy(top_n_index).cuda()
		top_n_prob=torch.gather(rpn_score,dim=1,index=top_n_index)
		part_imgs=torch.zeros([batch,self.topN,3,224,224]).cuda()
		for i in range(batch):
			for j in range(self.topN):
				[y0,x0,y1,x1]=top_n_cdds[i][j,1:5].astype(np.int)
				part_imgs[i:i+1,j]=F.interpolate(x_pad[i:i+1,:,y0:y1,x0:x1],size=(224,224),mode='bilinear',align_corners=True)
		part_imgs=part_imgs.view(batch*self.topN,3,224,224)

		_,_,part_features=self.pretrained_model(part_imgs.detach())
		part_feature=part_features.view(batch,self.topN,-1)

		part_feature=part_feature[:,:CAT_NUM,...].contiguous()
		part_feature=part_feature.view(batch,-1)

		concat_out=torch.cat([part_feature,feature],dim=1)
		# concat_out=concat_out.view(concat_out.size(0),(CAT_NUM+1),-1)
		# self.lstm1.flatten_parameters()
		# lstm1,hc=self.lstm1(concat_out)
		# inv_idx=torch.arange(lstm1.size(1)-1,-1,-1).long().cuda()
		# inv_lstm1=lstm1.index_select(1,inv_idx)
		# self.lstm2.flatten_parameters()
		# lstm2,_=self.lstm2(inv_lstm1,hc)
		# lstm2=lstm2.contiguous()
		# lstm2=lstm2.view(-1,lstm2.size(2))
		concat_logist=self.concat_net(concat_out)
		# concat_logist=concat_logist.view(x.size(0),(CAT_NUM+1),-1)
		raw_logist=resnet_out
		part_logits=self.partcls_net(part_features).view(batch,self.topN,-1)
		return [raw_logist,concat_logist,part_logits, top_n_index,top_n_prob,torch.from_numpy(top_n_cdds).cuda()]







