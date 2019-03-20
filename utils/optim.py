import torch.optim as optim

import numpy as np
__all__=['get_finetune_optimizer','reduce_lr_poly','get_optimizer_without_feature','get_adam','adjust_lr','reduce_lr'
		 ,"get_sgd"]


def get_finetune_optimizer(args, model):
	"""
	fine-tuning parameters with different lr
	:param args:
	:param model:
	:return: optimizer
	"""
	lr = args.lr
	weight_list = []
	bias_list = []
	last_weight_list = []
	last_bias_list = []
	for name, value in model.named_parameters():
		if 'cls' in name:
			print(name)
			if 'weight' in name:
				last_weight_list.append(value)
			elif 'bias' in name:
				last_bias_list.append(value)
		else:
			if 'weight' in name:
				weight_list.append(value)
			elif 'bias ' in name:
				bias_list.append(value)
	opt = optim.SGD([{'params': weight_list, 'lr': lr},
					 {'params': bias_list, 'lr': lr * 2},
					 {'params': last_weight_list, 'lr': lr * 10},
					 {'params': last_bias_list, 'lr': lr * 20}], momentum=0.9,
					weight_decay=1e-5, nesterov=True)
	return opt


def lr_poly(base_lr, iter, max_iter, power=0.9):
	return base_lr * ((1 - float(iter) / max_iter) ** (power))


def reduce_lr_poly(args, optimizer, iter, max_iter):
	"""

	:param args:
	:param optimizer:
	:param global_iter:
	:param max_iter:
	:return:
	"""
	base_lr = args.lr
	for g in optimizer.param_groups:
		g['lr'] = lr_poly(base_lr=base_lr, iter=iter, max_iter=max_iter, power=0.9)


def get_optimizer_without_feature(args, model):
	lr = args.lr
	opt = optim.Adam(params=[para for name, para in model.named_parameters()
							 if 'features' not in name], lr=lr, weight_decay=0.0001)
	return opt


def get_adam(args, model):
	lr = args.lr
	opt = optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.0005)
	return opt
def get_sgd(args,model):
	lr=args.lr
	opt=optim.SGD(params=model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
	return optim

def reduce_lr(args, optimizer, epoch, factor=0.1):
	"""
	decay in certain epoch
	:param args:  [.decay_points]
	:param optimizer:
	:param epoch:
	:param factor:
	:return:
	"""
	values = args.decay_points.strip().split(',')
	try:
		change_points = map(lambda x: int(x.strip()), values)
	except ValueError:
		change_points = None
	if change_points is not None and epoch in change_points:
		for g in optimizer.param_groups:
			g['lr'] = g['lr'] * factor
			print(epoch, g['lr'])
		return True


def adjust_lr(args, optimizer, epoch):
	"""
	decay lr in certain epoch
	:param args:  [.dataset] [.lr]
	:param optimizer:
	:param epoch:
	:return:
	"""
	if 'c9' in args.dataset:
		change_points = [25, 40]
	else:
		change_points = None
	if change_points is not None:
		change_points = np.array(change_points)
		pos = np.sum(epoch > change_points)
		lr = args.lr * (0.1 ** pos)
	else:
		lr = args.lr
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
