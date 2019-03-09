import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import os
import cv2
from torch.nn import LSTM
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import torch
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
		   'resnet152']

model_urls = {
	'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1x1(inplanes, planes)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = conv1x1(planes, planes * self.expansion)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,threshold=0.8,is_val=False):
		super(ResNet, self).__init__()
		self.threshold=threshold
		self.is_val=is_val
		self.inplanes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Linear(512 * block.expansion, num_classes)
		self.cls = self.classifier(512, num_classes)
		self.cls_erase = self.classifier(512, num_classes)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x,label=None):

		self.img_erased=x
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x= self.layer4(x)
		# feature map in the last layer
		self.map1=x
		# feat= self.avgpool(x)

		# flaten=feat.view(feat.size(0),-1)
		out=self.cls(x)
		log1 = self.avgpool(out).squeeze()
		if self.is_val:
			_,label=torch.max(log1,dim=1)
		self.attention= self.get_atten_map(out,label,True)
		feat_erase=self.erase_feature_maps(self.attention,x,self.threshold)

		out_erase=self.cls_erase(feat_erase)
		self.map_erase=out_erase

		log2=self.avgpool(out_erase).squeeze()

		return log1,log2

	def add_heatmap2img(self,img,heatmap):
		heatmap=heatmap*255
		color_map=cv2.applyColorMap(heatmap.astype(np.uint8),cv2.COLORMAP_JET)
		img_res=cv2.addWeighted(img.astype(np.uint8),0.5,color_map.astype(np.uint8),0.5,0)
		return img_res

	#TODO get the biger attention map
	def get_localization_maps(self):
		map1=self.normalize_atten_maps(self.map1)
		map_erase=self.normalize_atten_maps(self.map_erase)
		return torch.max(map1,map_erase)
	def get_heatmaps(self,gt_label):
		return self.get_atten_map(self.map1,gt_label)
	def get_fused_heatmap(self,gt_label):
		return self.get_heatmaps(gt_label)
	def get_maps(self,gt_label):
		return self.get_atten_map(self.map1,gt_label)

	def erase_feature_maps(self,atten_map_normed,feature_maps,threshold):

		if len(atten_map_normed.size())>3:
			atten_map_normed=torch.squeeze(atten_map_normed)
		atten_shape=atten_map_normed.size()
		pos=torch.ge(atten_map_normed,threshold)
		mask=torch.ones(atten_shape)
		mask[pos.data]=0.0
		mask=torch.unsqueeze(mask,dim=1)
		erased_feature_maps=feature_maps*mask.cuda()
		return erased_feature_maps

	def normalize_atten_maps(self,atten_maps):
		atten_shape=atten_maps.size()

		batch_mins,_ =torch.min(atten_maps.view(atten_shape[0:-2]+(-1,)),dim=-1,keepdim=True)
		batch_max, _= torch.max(atten_maps.view(atten_shape[0:-2]+(-1,)),dim=-1,keepdim=True)
		atten_normed=torch.div(atten_maps.view(atten_shape[0:-2]+(-1,))-batch_mins,batch_max-batch_mins)
		atten_normed=atten_normed.view(atten_shape)
		return atten_normed


	def saved_erased_img(self,img_path,img_batch=None):
		mean_vals=[]
		std_vals=[]
		if img_batch is None:
			img_batch = self.img_erased
		if len(img_batch.size())==4:
			batch_size=img_batch.size()[0]
			for batch_idx in range(batch_size):
				imgname=img_path[batch_idx]
				nameid=imgname.strip().split('/')[-1].strip().split('.')[0]
				atten_map=F.upsample(self.attention.unsqueeze(dim=1),(224,224),mode='bilinear')
				mask =atten_map.squeeze().cpu().data.numpy()

				img_dat=img_batch[batch_idx].cpu().data.numpy().transpose((1,2,0))
				img_dat=(img_dat*std_vals+mean_vals)*255
				#TODO size?
				mask=cv2.resize(mask,(321,321))
				img_dat=self.add_heatmap2img(img_dat,mask)
				save_path= os.path.join('../save_bins/',nameid+'.png')
				cv2.imwrite(save_path,img_dat)

	def get_atten_map(self,feature_maps,gt_label,normalize=True):
		label=gt_label.long()
		feature_map_size=feature_maps.size()
		batch_size=feature_map_size[0]
		atten_map=torch.zeros([batch_size,feature_map_size[2],feature_map_size[3]])
		atten_map=atten_map.cuda()
		for batch_idx in range(batch_size):
			atten_map[batch_idx,...]=torch.squeeze(feature_maps[batch_idx,label.data[batch_idx],...])
		if normalize:
			atten_map=self.normalize_atten_maps(atten_map)
		return atten_map


	def classifier(self, in_planes, out_planes):
		return nn.Sequential(
			# nn.Dropout(0.5),
			# nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
			# nn.ReLU(True),
			# nn.Dropout(0.5),
			# nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
			# nn.ReLU(True),
			nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0)  # fc8
		)

def resnet18(pretrained=False, **kwargs):
	"""Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        skip unmatched layer parameters (fc layer)
    """
	model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
	if pretrained:
		old_dict=model.state_dict()
		new_dict = model_zoo.load_url(model_urls['resnet18'])
		for k, v in new_dict.items():
			if k in old_dict.keys() and old_dict[k].size()==new_dict[k].size():
				old_dict[k]=v
		old_dict.update()
		model.load_state_dict(old_dict, strict=False)
	return model


def resnet34(pretrained=False, **kwargs):
	"""Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
	model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
	return model


def resnet50(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
	model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
	return model


def resnet101(pretrained=False, **kwargs):
	"""Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
	model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
	return model


def resnet152(pretrained=False, **kwargs):
	"""Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
	model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
	return model
