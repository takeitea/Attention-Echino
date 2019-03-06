import torch.nn as nn
import numpy as np

__all__ = ['weights_init_kaimingNorm', 'weights_init_kaimingUni','weights_init_xavierNormal','weights_init_xavierUniform']


def weights_init_kaimingUni(module):
	"""
	init the parameters in kaiming uniform
	:param module:
	:return:
	"""
	for m in module.modules:
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_uniform(m.weight, mode='fan_in', nonlinearity='relu')
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.uniform_(m.weight)
			nn.init.constant_(m.bias, 0.)
		elif isinstance(m, nn.Linear):
			nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
			if m.bias is not None:
				nn.init.constant_(m.bias, val=0.)


def weights_init_kaimingNorm(module):
	"""
	init the parameters in kaiming normal
	:param module:
	:return:
	"""
	for m in module.modules():
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.uniform_(m.weight)
			nn.init.constant_(m.weight, 0)
		elif isinstance(m, nn.Linear):
			nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)


def weights_init_xavierUniform(module):
	"""
	:param module:
	:return:
	"""
	for m in module.modules():
		if isinstance(m, nn.Conv2d):
			nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.uniform_(m.weight, a=0, b=1)
			nn.init.constant_(m.bias, val=0.)
		elif isinstance(m, nn.Linear):
			nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
			if m.bias is not None:
				nn.init.constant_(m.bias, val=0.)


def weights_init_xavierNormal(module):
	"""

	:param module:
	:return:
	"""
	for m in module.modules():
		if isinstance(m, nn.Conv2d):
			nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
			if m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.normal_(m.weight, 0, 0.01)
			nn.init.constant_(m.bias, val=0.)
		elif isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight, gain=np.sqrt(2))
			if m.bias is not None:
				nn.init.constant_(m.bias, val=0.)
