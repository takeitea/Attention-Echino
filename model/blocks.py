import torch.nn as nn
import torch
import torch.nn.functional as F

__all__= ['ConvBlock', 'ProjectorBlock', 'AttenBlock']


class ConvBlock(nn.Module):
	"""
	Conv Block of vgg net
	"""

	def __init__(self, in_dim, out_dim, nconv, stride=2, padding=1, bias=True, pool=False):
		super(ConvBlock, self).__init__()
		self.feature = [in_dim] + [out_dim for _ in range(nconv)]
		layer = []
		for i in range(len(self.feature) - 1):
			layer.append(nn.Conv2d(self.feature[i], self.feature[i + 1], kernel_size=3, padding=padding))
			layer.append(nn.BatchNorm2d(self.feature[i + 1]))
			layer.append(nn.ReLU())
			if pool:
				layer.append(nn.MaxPool2d(kernel_size=2, stride=stride))
		self.op = nn.Sequential(*layer)

	def forward(self, x):
		return self.op(x)


class ProjectorBlock(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(ProjectorBlock, self).__init__()
		self.op = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)

	def forward(self, x):
		return self.op(x)


class AttenBlock(nn.Module):
	def __init__(self, in_dim, normalize=True):
		super(AttenBlock, self).__init__()
		self.normalize = normalize
		self.op = nn.Conv2d(in_dim, out_channels=1, kernel_size=1)

	def forward(self, feat, filter_):
		N, C, W, H = feat.size()

		# TODO
		c = self.op(feat + filter_)
		if self.normalize:
			nor_filter = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
		else:
			nor_filter = torch.sigmoid(c)
		out = torch.mul(nor_filter.expand_as(feat), feat)
		if self.normalize:
			out = out.view(N, C, -1).sum(dim=2)
		else:
			out = F.adaptive_avg_pool2d(out, (1, 1)).view(N, C)
		return c.view(N, 1, W, H), out
