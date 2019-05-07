import torch
import os
from sklearn import metrics
import numpy as np
class AvgMeter(object):
	"""
	compute and store the avg and current value

	"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
	"""
	compute the precision@k for the value of k
	:param output:
	:param target:
	:param path:
	:param topk:
	:return:
	"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)
		_, pred = output.topk(maxk, 1)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1)).expand_as(pred)
		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul(100.0 / batch_size))
		return res
def save_test_txt(output,dir,path):
	pass

def accuracy_last_epoch(output, target, path, dir, topk=(1,)):
	"""
	compute the precision@k for the value of k
	:param output:
	:param target:
	:param path:
	:param topk:
	:return:
	"""
	with torch.no_grad():
		maxk = max(topk)
		batch_size = target.size(0)
		_, pred = output.topk(maxk, 1)
		pred = pred.t()
		if path:
			for i in range(batch_size) \
					:
				with open(os.path.join(dir, 'result.txt'), 'a') as L:
					L.writelines([path[i], ' ', str(pred[0, i].item()), '\n'])
		correct = pred.eq(target.view(1, -1)).expand_as(pred)
		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul(100.0 / batch_size))
		return res


def accuracy_lstm(output, target, dir, path, topk=(1,)):
	"""
	compute the precision@k for the value of k
	:param output:
	:param target:
	:param path:
	:param topk:
	:return:
	"""
	with torch.no_grad():
		output = output.view(output.size(0), output.size(1), -1)
		output = torch.mean(output, dim=1)
		maxk = max(topk)
		batch_size = target.size(0)
		_, pred = output.topk(maxk, 1)
		pred = pred.t()
		if path:
			for i in range(batch_size):
				with open(os.path.join(dir, 'result.txt'), 'a') as L:
					L.writelines([path[i], ' ', str(pred[0, i].item()), '\n'])
		correct = pred.eq(target.view(1, -1)).expand_as(pred)
		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
			res.append(correct_k.mul(100.0 / batch_size))
		return res




def get_mAP(gt_labels, pred_scores):
	n_classes = np.shape(gt_labels)[1]
	results = []
	for i in range(n_classes):
		res = metrics.average_precision_score(gt_labels[:, i], pred_scores[:, i])
		results.append(res)
	results = map(lambda x: '%.3f' % (x), results)
	cls_map = np.array(map(float, results))

	return cls_map


def get_AUC(gt_labels, pred_scores):
	res = metrics.roc_auc_score(gt_labels, pred_scores)
	return res


def _to_numpy(v):
	v = torch.squeeze(v)
	if torch.is_tensor(v):
		v = v.cpu()
		v = v.numpy()
	elif isinstance(v, torch.autograd.Variable):
		v = v.cpu().data.numpy()
	return v