import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.autograd import Variable

class BandPass(Function):
	def __init__(self):
		self.min=nn.Parameter(torch.randon())
		self.max=nn.parameter(torch.random())



class BandPassFunc(Function):
	@staticmethod
	def forward(ctx, feature,min,max,batch_size):
		ctx.save_for_backward(feature,min,max,batch_size)
		return torch.max(torch.min(feature,max),min)


	@staticmethod
	def backward(ctx, grad_outputs):

		pass
