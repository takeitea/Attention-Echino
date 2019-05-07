import torch
import torch.nn as nn
import torch.nn.functional as F



class SeparableConv2d(nn.Module):
	def __init__(self,in_channel,out_channel,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
		super(SeparableConv2d,self).__init__()
		self.conv=nn.Conv2d(in_channel,in_channel,kernel_size,stride,padding,dilation,in_channel,bias)
		self.pointwise=nn.Conv2d(in_channel,out_channel,1,1,0,1,1,bias)
	def forward(self, input):
		x=self.conv(input)
		x=self.pointwise(x)
		return x


class PreActBlock(nn.Module):
	'''Pre-activation version of the BasicBlock.'''
	expansion = 1
	def __init__(self, in_planes, planes, stride=1):
		super(PreActBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = SeparableConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = SeparableConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				SeparableConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
			)

	def forward(self, x):
		out = F.relu(self.bn1(x))
		shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
		out = self.conv1(out)
		out = self.conv2(F.relu(self.bn2(out)))
		out += shortcut
		return out


class PreActBottleneck(nn.Module):
	'''Pre-activation version of the original Bottleneck module.'''
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
		super(PreActBottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = SeparableConv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = SeparableConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes)
		self.conv3 = SeparableConv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				SeparableConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
			)

	def forward(self, x):
		out = F.relu(self.bn1(x))
		shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
		out = self.conv1(out)
		out = self.conv2(F.relu(self.bn2(out)))
		out = self.conv3(F.relu(self.bn3(out)))
		out += shortcut
		return out


class EchiNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=9, zero_init_residual=True):
		super(EchiNet, self).__init__()
		self.in_planes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
		# self.layer5=self._make_layer(block,1024,num_blocks[4],stride=2)
		self.linear = nn.Linear(1024 * block.expansion*4, num_classes)
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
				if isinstance(m, PreActBottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, PreActBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		# out = self.layer5(out)
		out = F.avg_pool2d(out, 7)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


def EchiNet_18(args):
	model= EchiNet(PreActBlock, [2, 2, 2, 2,2])
	old_dict=model.state_dict()
	if args.resume:
		new_dict=torch.load(args.resume)
		for k, v in new_dict.items():
			if k in old_dict.keys() and old_dict[k].size() == new_dict[k].size():
				old_dict[k] = v
		old_dict.update()
		model.load_state_dict(old_dict, strict=False)
	return model
def test():
	input=torch.randn([4,3,224,224])
	model=EchiNet_18()
	output=model(input)
	print(output)
# test()