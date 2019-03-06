import torch

class AvgMeter(object):
	"""
	compute and store the avg and current value

	"""
	def __init__(self):
		self.reset()
	def reset(self):
		self.val=0
		self.avg=0
		self.sum=0
		self.count=0

	def update(self,val,n=1):
		self.val=val
		self.sum+=val*n
		self.count+=n
		self.avg=self.sum/self.count


def accuracy( output,target,path,topk=(1,)):
	"""
	compute the precision@k for the value of k
	:param output:
	:param target:
	:param path:
	:param topk:
	:return:
	"""
	with torch.no_grad():
		maxk=max(topk)
		batch_size=target.size(0)
		_,pred=output.topk(maxk,1)
		pred=pred.t()
		if path :
			with open('result.txt','a') as L:
				L.writelines([path,pred,['\n']])
		correct=pred.eq(target.view(1,-1)).expand_as(pred)
		res=[]
		for k in topk:
			correct_k=correct[:k].view(-1).float().sum(0,keepdim=True)
			res.append(correct_k.mul(100.0/batch_size))
		return res



