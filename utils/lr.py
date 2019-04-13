import numpy as np
import matplotlib.pyplot as plt
import math

class Learning_rate_generater(object):
	"""
	Generate a list of learning rate
	"""

	def __init__(self, method, params, total_epoch,args):
		self.args = args
		if method == 'step':
			lr_factor, lr = self.step(params, total_epoch)
		elif method == 'log':
			lr_factor, lr = self.log(params, total_epoch)
		elif method == 'exp':
			lr_factor, lr = self.exp(params, total_epoch)
		elif method=='cos':
			lr_factor,lr=self.cos(params,total_epoch)
		else:
			raise KeyError('unknown method {}'.format(method))

		self.lr_factor = lr_factor
		self.lr = lr


	def step(self, params, total_epoch):
		lr_factor = []
		lr = []
		count = 0
		base_factor = 0.1
		for epoch in range(total_epoch):
			if count < len(params):
				if epoch >= params[count]:
					count += 1
			lr_factor.append(np.power(base_factor, count))
			lr.append( self.args.lr * lr_factor[epoch])
		return lr_factor, lr

	def log(self, params, total_epoch):
		min_, max_ = params[:2]
		np_lr = np.logspace(min_, max_, total_epoch)
		lr_factor = []
		lr = []
		for epoch in range(total_epoch):
			lr.append(np_lr[epoch])
			lr_factor.append(np_lr[epoch] / np_lr[0])
		if lr[0] !=self.args.lr:
			self.args.lr = lr[0]
		return lr_factor, lr

	def cos(self,min, total_epoch):
		lr_factor=[]
		lr=[]
		lr.append(self.args.lr)
		for epoch in range(total_epoch-1):
			cos_decay=0.5*(1+math.cos((epoch+1) *math.pi/total_epoch))
			decayed=(1-min)*cos_decay+min
			lr_factor.append(decayed)
			lr.append(self.args.lr*decayed)
		return lr_factor,lr
	def plot_lr(self):
		plt.figure(figsize=(8,8))
		plt.plot(np.arange(len(self.lr)),self.lr,color='blue',linewidth=2)
		plt.xlabel("epoch")
		plt.ylabel("lr")
		plt.title("lr cos anneal")
		plt.show()

	def adjust_learning_rate(self,optimizer, lr_factor, epoch):
		"""
		:param optimizer:
		:param lr_factor:
		:param epoch:
		:return:
		"""
		print('the lr is set to {0:.5f}'.format(lr_factor[epoch] * self.args.lr))
		for params_group in optimizer.param_groups:
			params_group['lr'] = lr_factor[epoch] *self.args.lr

	def cos_anneal_lr(self,optimizer, lr, epoch):
		"""
		:param optimizer:
		:param lr:
		:param epoch:
		:return:
		"""
		for params_group in optimizer.param_groups:
			params_group['lr'] = lr[epoch]