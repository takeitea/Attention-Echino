import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import sys

sys.path.append('../')
__all__ = ['model']

model_urls = {
	'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
	'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
	'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
	'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
	'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
	'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
	'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
	'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}




class VGG(nn.Module):
	def __init__(self,cfg,  num_classes=9, args=None, threshold=None):
		super(VGG, self).__init__()
		self.block1=make_layers(cfg['B1'],in_channels=3)
		self.block2=make_layers(cfg['B2'],in_channels=64)
		self.block3=make_layers(cfg['B3'],in_channels=128)
		self.block4=make_layers(cfg['B4'],in_channels=256)
		self.block5=make_layers(cfg['B5'],in_channels=512)
		self.b3_conv1=nn.Conv2d(256,64,1)
		self.b3_conv2=nn.Conv2d(64,64,3,padding=1)
		self.b3_conv3=nn.Conv2d(64,2,1)
		self.b4_conv1=nn.Conv2d(512,128,1)
		self.b4_conv2=nn.Conv2d(128,128,3,padding=1)
		self.b4_conv3=nn.Conv2d(128,2,1)
		self.cls = self.classifier(512, num_classes)

		self._initialize_weights()
		self.onehot = args.onehot
		self.mask_loss=nn.BCEWithLogitsLoss()
		if args is not None and args.onehot == 'True':
			self.loss_cross_entropy = nn.BCEWithLogitsLoss()
		else:
			self.loss_cross_entropy = nn.CrossEntropyLoss()

	def classifier(self, in_planes, out_planes):
		return nn.Sequential(
			nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1, dilation=1),
			nn.ReLU(True),
			nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),
			nn.ReLU(True),
			nn.Conv2d(1024, out_planes, kernel_size=1, padding=0))

	def forward(self, x, label=None):
		b1=self.block1(x)
		b2=self.block2(b1)
		b3=self.block3(b2)
		b3_conv1=self.b3_conv1(b3)
		b3_conv2=self.b3_conv2(b3_conv1)
		b3_conv3=self.b3_conv3(b3_conv2)

		b4=self.block4(b3)
		b4_conv1=self.b4_conv1(b4)
		b4_conv2=self.b4_conv2(b4_conv1)
		b4_conv3=self.b4_conv3(b4_conv2)
		b5=self.block5(b4)
		out = self.cls(b5)
		self.map1 = out
		logits_1=torch.mean(torch.mean(out,dim=2),dim=2)
		return [logits_1,b3_conv3,b4_conv3]

	def get_weight(self, gt):
		pos_num = torch.sum(gt)
		neg_num = torch.sum(1-gt)
		total = pos_num + neg_num
		return gt


	def get_loss(self, logits, gt_labels,mask1):
		if self.onehot == 'True':
			gt = gt_labels.float()
		else:
			gt = gt_labels.long()
		mask1=mask1.float()
		loss_cls = self.loss_cross_entropy(logits[0], gt)
		# loss_mask1=self.mask_loss(logits[1], mask1)
		# loss_mask2=self.mask_loss(logits[2], mask1)
		loss_mask1 = F.binary_cross_entropy_with_logits(logits[1], mask1, weight=0.4)
		loss_mask2 = F.binary_cross_entropy_with_logits(logits[2], mask1, weight=0.6)
		loss_mask=loss_mask1+loss_mask2
		return [loss_cls, loss_mask]

	def get_localization_maps(self):
		return self.normalize_atten_maps(self.map1)

	def get_heatmaps(self, gt_label):
		map1 = self.get_atten_map(self.map1, gt_label)
		return [map1, ]

	def get_fused_heatmap(self, gt_label):
		maps = self.get_heatmaps(gt_label=gt_label)
		fuse_atten = maps[0]
		return fuse_atten

	def get_maps(self, gt_label):
		map1 = self.get_atten_map(self.map1, gt_label)
		return [map1, ]

	def normalize_atten_maps(self, atten_maps):
		atten_shape = atten_maps.size()

		batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
		batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
		atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,)) - batch_mins,
								 batch_maxs - batch_mins)
		atten_normed = atten_normed.view(atten_shape)
		return atten_normed

	def get_atten_map(self, feature_maps, gt_labels, normalize=True):
		label = gt_labels.long()
		feature_map_size = feature_maps.size()
		batch_size = feature_map_size[0]
		atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
		atten_map = Variable(atten_map.cuda())
		for batch_idx in range(batch_size):
			atten_map[batch_idx, :, :] = torch.squeeze(feature_maps[batch_idx, label.data[batch_idx], :, :])
		if normalize:
			atten_map = self.normalize_atten_maps(atten_map)
		return atten_map

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()


def make_layers(cfg,in_channels, batch_norm=False):
	layers = []
	for v in cfg :
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
		elif v == 'N':
			layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, dilation=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	return nn.Sequential(*layers)


cfg = {
	'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
	'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}
cfg_mask = {'B1': [64, 64, 'M'],
			'B2': [128, 128, 'M'],
			'B3': [256, 256, 256, 'M'],
			'B4': [512, 512, 512, 'N'],
			'B5': [512, 512, 512, 'N']}



def model(pretrained=False, **kwargs):
	""" VGG 16-layer model """
	model = VGG(cfg_mask, **kwargs)

	if pretrained:

		model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
	return model
