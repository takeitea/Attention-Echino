import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from model import AttenBlock,ProjectorBlock,ConvBlock,weights_init_kaimingNorm,weights_init_xavierUniform
'''
attention model in vgg net
'''

class AttenVgg(nn.Module):
	def __init__(self,input_size,num_class,attention=True,normalize_atten=True,init='xavierUniform'):
		super(AttenVgg,self).__init__()
		self.atten=attention
		self.conv1=ConvBlock(3,64,2)
		self.conv2=ConvBlock(64,128,2)
		self.conv3=ConvBlock(128,256,3)
		self.conv4=ConvBlock(256,512,3)

		self.conv5=ConvBlock(512,512,3)
		self.conv6=ConvBlock(512,512,2,pool=True)
		self.maxpool=nn.MaxPool2d(kernel_size=2,stride=2)
		self.dense=nn.Conv2d(512,512,kernel_size=int(input_size/32))

		# Projectors
		if self.atten:
			self.projection=ProjectorBlock(256,512)
			self.atten1=AttenBlock(512,normalize=normalize_atten)
			self.atten2=AttenBlock(512,normalize=normalize_atten)
			self.atten3=AttenBlock(512,normalize=normalize_atten)

		if self.atten:
			self.cls=nn.Linear(512*3,num_class)
		else:
			self.cls=nn.Linear(512,num_class)

		if init=='kaimingNorm':

			weights_init_kaimingNorm(self)
		else:
			weights_init_xavierUniform(self)



	def forward(self, x):
		feat_1=self.conv1(x)
		feat_2=self.conv2(feat_1)
		feat_3=self.conv3(feat_2)
		feat_3_p=self.maxpool(feat_3)
		feat_4=self.conv4(feat_3_p)
		feat_4_p=self.maxpool(feat_4)
		feat_5=self.conv5(feat_4_p)
		feat_5_p=self.maxpool(feat_5)
		feat_6=self.conv6(feat_5_p)
		glob=self.dense(feat_6)

		if self.atten:
			c1,atten_feat_1=self.atten1(self.projection(feat_3_p),glob)
			c2,atten_feat_2=self.atten2(feat_4_p,glob)
			c3,atten_feat_3=self.atten3(feat_5_p,glob)
			out=self.cls(torch.cat((atten_feat_1,atten_feat_2,atten_feat_3),dim=1))
		else:
			c1,c2,c3=None,None,None
			out=self.cls(torch.squeeze(glob))
		return out,c1,c2,c3
