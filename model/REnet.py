from torch import nn
from .resnet import resnet18
from model import drn_c_26
import torch
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn.functional as F
class REnet(nn.Module):
	def __init__(self,num_classes=9):
		super(REnet,self).__init__()
		self.pretrained_model=ptcv_get_model( "darknet53",pretrained=False)
		self.pretrained_model.load_state_dict(torch.load("./model/darknet53.pth"))

		# self.pretrained_model=drn_c_26(pretrained=True,out_map=True)
		self.num_classes=num_classes
		self.lstm1=nn.LSTM(1024,128,batch_first=True,num_layers=1,bidirectional=False,bias=False)
		self.lstm2=nn.LSTM(128,128,batch_first=True,num_layers=1,bidirectional=False,bias=False)
		self.fc=nn.Linear(128,self.num_classes)
		self.n_part=5

	def forward(self, x):

		# main_out,conv_feature,main_feature=self.pretrained_model(x[0])
		batch=x.size(0)
		# log_pre,_,_ =self.pretrained_model(x[:,-1,...])
		# _,class_pre=torch.max(log_pre,dim=1)

		x=x.view(-1,x.size(2),x.size(3),x.size(4))
		part_features,_=self.pretrained_model(x)
		part_feature=part_features.view(batch,self.n_part,-1)
		self.lstm1.flatten_parameters()
		lstm1,hc=self.lstm1(part_feature )
		inv_idx=torch.arange(lstm1.size(1)-1,-1,-1).long().cuda()
		inv_lstm1=lstm1.index_select(1,inv_idx)
		self.lstm2.flatten_parameters()
		lstm2,_=self.lstm2(inv_lstm1,hc)
		lstm2=lstm2.contiguous()

		lstm2=lstm2.view(-1,lstm2.size(2))
		lstm2_out=self.fc(lstm2)
		lstm2_out=lstm2_out.view(batch,self.n_part,-1)
		# lstm2_out=F.log_softmax(lstm2_out,dim=2)
		return lstm2_out






