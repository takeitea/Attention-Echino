import numpy as np
import torch.nn.functional as F
import os
import torch

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import cv2
BatchNorm = nn.BatchNorm2d


webroot = 'https://tigress-web.princeton.edu/~fy/drn/models/'

model_urls = {
	'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'drn-c-26': webroot + 'drn_c_26-ddedf421.pth',
	'drn-c-42': webroot + 'drn_c_42-9d336e8c.pth',
	'drn-c-58': webroot + 'drn_c_58-0a53a92c.pth',
	'drn-d-22': webroot + 'drn_d_22-4bd2f8ea.pth',
	'drn-d-38': webroot + 'drn_d_38-eebb45f0.pth',
	'drn-d-54': webroot + 'drn_d_54-0e0534ff.pth',
	'drn-d-105': webroot + 'drn_d_105-12b40979.pth'
}


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=padding, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None,
				 dilation=(1, 1), residual=True):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride,
							 padding=dilation[0], dilation=dilation[0])
		self.bn1 = BatchNorm(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes,
							 padding=dilation[1], dilation=dilation[1])
		self.bn2 = BatchNorm(planes)
		self.downsample = downsample
		self.stride = stride
		self.residual = residual

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)
		if self.residual:
			out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None,
				 dilation=(1, 1), residual=True):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn1 = BatchNorm(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
							   padding=dilation[1], bias=False,
							   dilation=dilation[1])
		self.bn2 = BatchNorm(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = BatchNorm(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class DRN(nn.Module):

	def __init__(self, block, layers, num_classes=9,
				 channels=(16, 32, 64, 128, 256, 512, 512, 512),
				 out_map=False, out_middle=False, arch='D',threshold=0.5):
		super(DRN, self).__init__()
		self.inplanes = channels[0]
		self.out_map = out_map
		self.out_dim = channels[-1]
		self.out_middle = out_middle
		self.arch = arch
		self.threshold=threshold
		# self.sigmoid=nn.Sigmoid()
		self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
							   padding=3, bias=False)
		self.bn1 = BatchNorm(channels[0])
		self.relu = nn.ReLU(inplace=True)
		self.atten=nn.AdaptiveAvgPool2d((28,28))
		self.layer1 = self._make_layer(
			BasicBlock, channels[0], layers[0], stride=1)
		self.layer2 = self._make_layer(
			BasicBlock, channels[1], layers[1], stride=2)

		self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)

		self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
		self.layer5 = self._make_layer(block, channels[4], layers[4],
									   dilation=2, new_level=False)
		self.layer6 = None if layers[5] == 0 else \
			self._make_layer(block, channels[5], layers[5], dilation=4,
							 new_level=False)

		self.layer7 = None if layers[6] == 0 else \
			self._make_layer(BasicBlock, channels[6], layers[6], dilation=2,
							 new_level=False, residual=False)
		self.layer8 = None if layers[7] == 0 else \
			self._make_layer(BasicBlock, channels[7], layers[7], dilation=1,
							 new_level=False, residual=False)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = nn.Conv2d(self.out_dim, num_classes, kernel_size=1,
							stride=1, padding=0, bias=True)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, BatchNorm):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def get_localization_maps(self):
		map1 = self.normalize_atten_maps(self.map1)
		map_erase = self.normalize_atten_maps(self.map_erase)
		return torch.max(map1, map_erase)

	def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
					new_level=True, residual=True):
		assert dilation == 1 or dilation % 2 == 0
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				BatchNorm(planes * block.expansion),
			)

		layers = list()
		layers.append(block(
			self.inplanes, planes, stride, downsample,
			dilation=(1, 1) if dilation == 1 else (
				dilation // 2 if new_level else dilation, dilation),
			residual=residual))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, residual=residual,
								dilation=(dilation, dilation)))
		return nn.Sequential(*layers)


	def _make_conv_layers(self, channels, convs, stride=1, dilation=1):
		modules = []
		for i in range(convs):
			modules.extend([
				nn.Conv2d(self.inplanes, channels, kernel_size=3,
						  stride=stride if i == 0 else 1,
						  padding=dilation, bias=False, dilation=dilation),
				BatchNorm(channels),
				nn.ReLU(inplace=True)])
			self.inplanes = channels
		return nn.Sequential(*modules)

	def forward(self, x):
		self.img_erased=x
		y = list()
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.layer1(x)
		y.append(x)
		x = self.layer2(x)
		y.append(x)
		x = self.layer3(x)
		y.append(x)
		x = self.layer4(x)
		y.append(x)
		x = self.layer5(x)
		y.append(x)
		if self.layer6 is not None:
			x = self.layer6(x)
			y.append(x)

		if self.layer7 is not None:
			x = self.layer7(x)
			y.append(x)

		if self.layer8 is not None:
			x = self.layer8(x)
			y.append(x)

		if self.out_map:
			x = self.fc(x)
		else:

			map1 = self.fc(x)
			self.map1 = map1

			out= self.avgpool(x)
			log1 = out.view(x.size(0), -1)

			_,label=torch.max(log1,dim=1)
			self.attention=self.get_atten_map(x,label,True)

			feat_erase=self.erase_feature_maps(self.attention,x,self.threshold)
			out_erase=self.fc(feat_erase)
			self.map_erase=out_erase
			log2=self.avgpool(out_erase).view(x.size(0),-1)
			# fused_map=self.get_fused_maps()
			#
			# log3=self.avgpool(self.fc(fused_map*x)).view(x.size(0),-1)
			return log1,log2

		if self.out_middle:
			return x, y
		else:
			return x
	def get_fused_heatmap(self,gt_label):
		return self.get_heatmaps(gt_label)
	def saved_erased_img(self,img_path,img_batch=None):
		mean_vals= [0.275, 0.278, 0.284],
		std_vals = [0.170, 0.171, 0.173]
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
				mask=cv2.resize(mask,(224,224))
				img_dat=self.add_heatmap2img(img_dat,mask)
				save_path= os.path.join('../save_bins/',nameid+'.png')
				cv2.imwrite(save_path,img_dat)
	def add_heatmap2img(self,img,heatmap):
		heatmap=heatmap*255
		color_map=cv2.applyColorMap(heatmap.astype(np.uint8),cv2.COLORMAP_JET)
		img_res=cv2.addWeighted(img.astype(np.uint8),0.5,color_map.astype(np.uint8),0.5,0)
		return img_res
	def erase_feature_maps(self,atten_map_normed,feature_maps,threshold):
		if len(atten_map_normed.size())>3:
			atten_map_normed=torch.squeeze(atten_map_normed)
		atten_shape=atten_map_normed.size()
		pos=torch.ge(atten_map_normed,threshold)
		mask=torch.ones(atten_shape)
		mask[pos.data]=0.
		mask=torch.unsqueeze(mask,dim=1)
		erased_feature_maps=feature_maps*mask.cuda()
		return erased_feature_maps

	def get_fused_maps(self):
		map1=self.normalize_atten_maps(self.map1)
		map_erase=self.normalize_atten_maps(self.map_erase)
		return torch.max(map1,map_erase)
	def get_atten_map(self,feature_maps,gt_label,normalize=True):
		label=gt_label.long()
		feature_map_size=feature_maps.size()
		batch_size=feature_map_size[0]
		atten_map=torch.zeros([batch_size,feature_map_size[2],feature_map_size[3]])

		atten_map=atten_map.cuda()
		for batch_idx in range(batch_size):
			atten_map[batch_idx,...]=torch.squeeze(feature_maps[batch_idx,label.data[batch_idx,...]])
		if normalize:
			atten_map=self.normalize_atten_maps(atten_map)
		return atten_map
	def normalize_atten_maps(self,atten_map):
		atten_shape=atten_map.size()

		batch_mins,_=torch.min(atten_map.view(atten_shape[0:-2]+(-1,)),dim=-1,keepdim=True)
		batch_max,_=torch.max(atten_map.view(atten_shape[0:-2]+(-1,)),dim=-1,keepdim=True)
		atten_normed=torch.div(atten_map.view(atten_shape[0:-2]+(-1,))-batch_mins,batch_max-batch_mins)
		atten_normed=atten_normed.view(atten_shape)
		return atten_normed
class DRN_A(nn.Module):

	def __init__(self, block, layers, num_classes=1000):
		self.inplanes = 64
		super(DRN_A, self).__init__()
		self.out_dim = 512 * block.expansion
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
									   dilation=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
									   dilation=4)
		self.avgpool = nn.AvgPool2d(28, stride=1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, BatchNorm):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	# for m in self.modules():
	#     if isinstance(m, nn.Conv2d):
	#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
	#     elif isinstance(m, nn.BatchNorm2d):
	#         nn.init.constant_(m.weight, 1)
	#         nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes,
								dilation=(dilation, dilation)))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


def drn_a_50(pretrained=False, **kwargs):
	model = DRN_A(Bottleneck, [3, 4, 6, 3], **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
	return model


def drn_c_26(pretrained=False, **kwargs):
	model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
	if pretrained:
		# model.load_state_dict(model_zoo.load_url(model_urls['drn-c-26']), strict=False)
		old_dict = model.state_dict()
		new_dict = model_zoo.load_url(model_urls['drn-c-26'])
		for k, v in new_dict.items():
			if k in old_dict.keys() and old_dict[k].size() == new_dict[k].size():
				old_dict[k] = v
		old_dict.update()
		model.load_state_dict(old_dict, strict=False)
	return model


def drn_c_42(pretrained=False, **kwargs):
	model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-c-42']))
	return model


def drn_c_58(pretrained=False, **kwargs):
	model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-c-58']))
	return model


def drn_d_22(pretrained=False, **kwargs):
	model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-22']))
	return model


def drn_d_24(pretrained=False, **kwargs):
	model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-24']))
	return model


def drn_d_38(pretrained=False, **kwargs):
	model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-38']))
	return model


def drn_d_40(pretrained=False, **kwargs):
	model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-40']))
	return model


def drn_d_54(pretrained=False, **kwargs):
	model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-54']))
	return model


def drn_d_56(pretrained=False, **kwargs):
	model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-56']))
	return model


def drn_d_105(pretrained=False, **kwargs):
	model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-105']))
	return model


def drn_d_107(pretrained=False, **kwargs):
	model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 2, 2], arch='D', **kwargs)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['drn-d-107']))
	return model
