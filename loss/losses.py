from torch.autograd import Variable
import torch.nn as nn
from torch.nn import functional as F
import torch


class ClusteringAffinity(nn.Module):
	def __init__(self, n_classes, n_centers, sigma, feat_dim, init_weight=True, **kwargs):
		super(ClusteringAffinity, self).__init__()
		self.n_classes = n_classes
		self.n_centers = n_centers
		self.feat_dim = feat_dim
		self.sigma = sigma
		self.centers = nn.Parameter(torch.randn(self.n_classes, self.n_centers, self.feat_dim))
		if init_weight:
			self.__init_weight()

	def __init_weight(self):
		nn.init.kaiming_normal_(self.centers)

	def forward(self, f):
		f_expand = f.unsqueeze(1)
		w_expand = self.centers.unsqueeze(0)
		fw_norm = torch.sum((f_expand - w_expand) ** 2, -1)
		distance = torch.exp(-fw_norm / self.sigma)
		distance = torch.max(distance, -1)

		# Regularization
		mc = self.n_centers * self.n_classes
		w_reshape = self.centers.view(mc, self.feat_dim)
		w_reshape_expand1 = w_reshape.unsqueeze(0)
		w_reshape_expand2 = w_reshape.unsqueeze(1)
		w_norm_mat = torch.sum((w_reshape_expand2 - w_reshape_expand1) ** 2, -1)
		w_norm_upper = torch.triu(w_norm_mat)
		mu = 2.0 / (mc ** 2 - mc) * w_norm_upper.sum()
		residuals = ((w_norm_upper - mu) ** 2).triu()
		rw = 2.0 / (mc ** 2 - mc) * residuals.sum()
		batch_size = f.size(0)
		rw_broadcast = torch.ones((batch_size, 1)) * rw
		output = torch.cat((distance, rw_broadcast), -1)
		return output


class Affinity_Loss(nn.Module):
	def __init__(self, lambd):
		super(Affinity_Loss, self).__init__()
		self.lamda = lambd

	def forward(self, y_tru_pluseone, y_pred_plusone):
		onehot = y_tru_pluseone[:, :-1]
		distance = y_pred_plusone[:, :-1]
		rw = torch.mean(y_pred_plusone[:, -1])
		d_fi_wyi = torch.sum(onehot * distance, -1, keepdims=True)
		losses = torch.clam(self.lamda + distance - d_fi_wyi, min=0)
		L_mm = torch.sum(losses * (1 - onehot), -1)
		return L_mm + rw


class MultiLoss(nn.CrossEntropyLoss):
	def __init__(self, ):
		super(MultiLoss, self).__init__(None, reduction='elementwise_mean')

	def forward(self, output, target):
		loss = torch.autograd.Variable(torch.zeros(1), ).cuda()

		output_2 = torch.chunk(output, 2, dim=-1)
		for output_ in output_2:
			parts = torch.chunk(output_, 5, dim=1)
			for part in parts:
				loss += F.cross_entropy(part.squeeze(1), target)
		return loss


class HEM_Loss(nn.CrossEntropyLoss):
	""""
	Online hard example mining
	"""

	def __init__(self, ratio):
		super(HEM_Loss, self).__init__(None, reduction='elementwise_mean')
		self.ratio = ratio

	def forward(self, output, target, ratio=None):
		if ratio is not None:
			self.ratio = ratio
		batch_size = output.size(0)
		num_hardsample = int(self.ratio * batch_size)
		output_ = output.clone()
		instance_losses = torch.autograd.Variable(torch.zeros(batch_size)).cuda()
		for idx, label in enumerate(target.data):
			instance_losses[idx] = -output_.data[idx, label]

		_, idxs = instance_losses.topk(num_hardsample)
		output_hardsample = output.index_select(0, idxs)
		target_hardsample = target.index_select(0, idxs)
		return F.cross_entropy(output_hardsample, target_hardsample)


class Auxiliary_Loss(nn.CrossEntropyLoss):
	"""
	TO calculate the  top-2 k loss
	which means that the most difficult  categories
	"""

	def __init__(self):
		super(Auxiliary_Loss, self).__init__()

	def forward(self, output, target):
		_,pred=output.topk(2,1)
		pred=pred.t()
		idxs=[]
		for inx,label in enumerate(target.data):
			if label in pred[:,inx]:
				idxs.append(inx)
		idxs=torch.Tensor(idxs).cuda().long()
		if idxs is []:
			return Variable(torch.zeros(1).cuda())
		output_hard=output.index_select(0,idxs)
		target_hard=target.index_select(0,idxs)
		return F.cross_entropy(output_hard,target_hard)

def list_loss(logits, targets):
	temp = F.log_softmax(logits, -1)
	loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]
	return torch.stack(loss)


PROPOSAL_NUM = 6


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
	loss = Variable(torch.zeros(1).cuda())
	batch_size = score.size(0)
	for i in range(proposal_num):
		targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
		pivot = score[:, i].unsqueeze(1)
		loss_p = (1 - pivot + score) * targets_p
		loss_p = torch.sum(F.relu(loss_p))
		loss += loss_p
	return loss / batch_size


kl = nn.KLDivLoss()


def KL_Loss(raw_logits, part_logits, target, tmp):
	batch_size = target.size(0)
	kl_loss = torch.zeros(1).cuda()
	for i in range(part_logits.size(1)):
		kl_loss += kl(F.softmax(raw_logits / tmp, dim=1), F.softmax(part_logits[:, i, :] / tmp, dim=1)) * tmp ** 2
	return kl_loss / batch_size / part_logits.size(1)
