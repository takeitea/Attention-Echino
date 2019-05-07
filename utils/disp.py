import numpy as np
import shutil
import cv2
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from .torchviz import make_dot
import os
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
	print('no display found.')
	mpl.use("Agg")
import matplotlib.pyplot as plt
import scipy.io as sio

__all__ = ['visualize_atten_sigmoid', 'save_checkpoint', 'visualize_atten_softmax', 'vizNet', 'Stats', 'plot_curve']


def visualize_atten_softmax(Img, c, gain, nrow):
	image = Img.permute((1, 2, 0)).cpu().numpy()

	# heatmap
	N, C, W, H = c.size()
	a = F.softmax(c.view(N, C, -1), dim=2).view(N, C, W, H)
	if gain > 1:
		a = F.interpolate(a, scale_factor=gain, mode='bilinear')
	atten = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True)
	atten = atten.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
	atten = cv2.applyColorMap(atten, cv2.COLORMAP_JET)
	atten = cv2.cvtColor(atten, cv2.COLOR_BGR2RGB)
	atten = np.float32(atten) / 255
	vis = 0.6 * image + 0.4 * atten
	return torch.from_numpy(vis).permute(2, 0, 1)


def visualize_atten_sigmoid(Img, c, gain, nrow):
	image = Img.permute((1, 2, 0)).cpu().numpy()
	a = F.sigmoid(c)
	if gain > 1:
		a = F.interpolate(a, scale_factor=gain, mode='bilinear')
	atten = utils.make_grid(a, nrow=nrow, normalize=False)
	atten = atten.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
	atten = cv2.applyColorMap(atten, cv2.COLORMAP_JET)
	atten = cv2.cvtColor(atten, cv2.COLOR_BGR2RGB)
	atten = np.float32(atten) / 255
	vis = 0.6 * image + 0.4 * atten
	return torch.from_numpy(vis).permute(2, 0, 1)


def vizNet(model, path):
	x = torch.randn(10, 3, 224, 224)
	y = model(x)
	g = make_dot(y[0])
	g.render(os.path.join(path, 'graph'), view=False)


class Stats:
	def __init__(self, path, start_epoch, total_epoch):
		if start_epoch is not 0:
			stats_ = sio.loadmat(os.path.join(path, 'stats.mat'))
			data = stats_['data']
			content = data[0, 0]

			self.trainObj = content['trainObj'][:, :start_epoch].squeeze().tolist()
			self.trainTop1 = content['trainTop1'][:, :start_epoch].squeeze().tolist()
			self.trainTop2 = content['trainTop2'][:, :start_epoch].squeeze().tolist()
			self.valobj = content['valObj'][:, :start_epoch].squeeze().tolist()
			self.valTop1 = content['valTop1'][:, :start_epoch].squeeze().tolist()
			self.valTop2 = content['valTop2'][:, :start_epoch].squeeze().tolist()
			if start_epoch is 1:
				self.trainObj = [self.trainObj]
				self.trainTop1 = [self.trainTop1]
				self.trainTop2 = [self.trainTop2]
				self.valobj = [self.valobj]
				self.valTop1 = [self.valTop1]
				self.valTop2 = [self.valTop2]
		else:
			self.trainObj = []
			self.trainTop1 = []
			self.trainTop2 = []
			self.valobj = []
			self.valTop1 = []
			self.valTop2 = []
		self.total_epoch = total_epoch

	def _update(self, trainObj, top1, top2, valobj, prec1, prec2):
		self.trainObj.append(trainObj)
		self.trainTop1.append(top1.cpu().numpy())
		self.trainTop2.append(top2.cpu().numpy())
		self.valobj.append(valobj)
		self.valTop1.append(prec1.cpu().numpy())
		self.valTop2.append(prec2.cpu().numpy())

	def get_lastn(self,n=5):

		self.last5prec1=np.mean(self.valTop1[self.total_epoch - n:self.total_epoch])
		self.last5prec2=np.mean(self.valTop2[self.total_epoch - n:self.total_epoch])
		print("last 5 epochs mean top1 {}".format(self.last5prec1))
		print("last l epochs mean top2 {}".format(self.last5prec2))


def plot_curve(stats, path, iserr):
	trainObj = np.array(stats.trainObj)
	valobj = np.array(stats.valobj)
	if iserr:
		trainTop1 = 100 - np.array(stats.trainTop1)
		trainTop2 = 100 - np.array(stats.trainTop2)
		valTop1 = 100 - np.array(stats.valTop1)
		valTop2 = 100 - np.array(stats.valTop2)
		title = 'error'
	else:
		trainTop1 = np.array(stats.trainTop1)
		trainTop2 = np.array(stats.trainTop2)
		valTop1 = np.array(stats.valTop1)
		valTop2 = np.array(stats.valTop2)
		title = 'accuracy'
	epoch = len(trainObj)
	figure = plt.figure()
	obj = plt.subplot(1, 3, 1)
	obj.plot(range(1, epoch + 1), trainObj, 'o-', label='train')
	obj.plot(range(1, epoch + 1), valobj, 'o-', label='val')
	plt.xlabel('epoch')
	plt.title('objective')
	handles, labels = obj.get_legend_handles_labels()
	obj.legend(handles[::-1], labels[::-1])
	top1 = plt.subplot(1, 3, 2)
	top1.plot(range(1, epoch + 1), trainTop1, 'o-', label='train')
	top1.plot(range(1, epoch + 1), valTop1, 'o-', label='val')
	plt.title('top1' + title)
	plt.xlabel('epoch')
	handles, labels = top1.get_legend_handles_labels()
	top1.legend(handles[::-1], labels[::-1])
	top2 = plt.subplot(1, 3, 3)
	top2.plot(range(1, epoch + 1), trainTop2, 'o-', label='train')
	top2.plot(range(1, epoch + 1), valTop2, 'o-', label='val')
	plt.title('top5' + title)
	plt.xlabel('epoch')
	handles, labels = top2.get_legend_handles_labels()
	top2.legend(handles[::-1], labels[::-1])
	filename = os.path.join(path, 'net-train.pdf')
	figure.savefig(filename, bbox_inches='tight')
	plt.close()


def save_checkpoint(state, is_best, filename):
	torch.save(state, filename[0])
	if is_best:
		shutil.copyfile(filename[0], filename[1])
