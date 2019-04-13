import argparse
import torch
import time
import numpy as np
import torch.optim as optim
import os
import scipy.io as sio
from model import drn_c_26_mask
from utils import AvgMeter, accuracy, plot_curve, restore
from utils import vizNet, Stats, save_checkpoint, loadpartweight
from data import get_with_mask
import shutil
import torch.nn.functional  as F
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
best_prec1 = 0


def arg_pare():
	arg = argparse.ArgumentParser(description=" args of resnet18")
	arg.add_argument('-bs', '--batch_size', help='batch size', default=40)
	arg.add_argument('--store_per_epoch', default=False)
	arg.add_argument('--epochs', default=55)
	arg.add_argument('--num_classes', default=2, type=int)
	arg.add_argument('--lr', help='learn rate', default=0.001)
	arg.add_argument('-att', '--attention', help='whether to use attention', default=True)
	arg.add_argument('--img_size', help='the input size', default=224)
	arg.add_argument('--dir', help='the dataset root', default='./datafolder/ROI/image/')
	arg.add_argument('--print_freq', default=180, help='the frequency of print infor')
	arg.add_argument('--modeldir', help=' the model viz dir ', default='drn_mask')
	arg.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='# of workers')
	arg.add_argument('--lr_method', help='method of learn rate')
	arg.add_argument('--gpu', default=4, type=str)
	arg.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
	arg.add_argument('--dist_url', default='tcp://127.0.0.01:123', type=str, help='url used to set up')
	arg.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend')
	arg.add_argument('--evaluate', default=False, help='whether to evaluate only')
	arg.add_argument('--mean5', default=35, help="the first epoch to calculate the 5-epoch means")
	arg.add_argument('--start_epoch', default=0)
	# arg.add_argument('--resume', default='./ResNet18_mask/checkpoint.pth.tar', help="whether to load checkpoint")
	arg.add_argument('--resume', default=False, help="whether to load checkpoint")
	return arg.parse_args()


args = arg_pare()


def main():
	print('\n loading the dataset ... \n')
	print('\n done \n')
	model =drn_c_26_mask(num_classes=1).cuda()
	LR = Learning_rate_generater('step', [30, 40], args.epochs)
	opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.90, weight_decay=1e-4)
	print(args)
	# plot network
	# vizNet(model, args.modeldir)
	if args.resume:
		restore(args, model, opt, istrain=not args.evaluate)
	model = torch.nn.DataParallel(model, range(args.gpu))
	trainloader, valloader =get_with_mask(args)
	critertion = torch.nn.BCELoss().cuda()
	if args.evaluate:
		evaluate(valloader, model, critertion)
		return
	if os.path.exists(args.modeldir):
		shutil.rmtree(args.modeldir)

	os.mkdir(args.modeldir)

	stats = Stats(args.modeldir, start_epoch=args.start_epoch, total_epoch=args.epochs)
	for epoch in range(args.epochs):

		is_last=epoch==args.epochs-1

		adjust_learning_rate(opt, LR.lr_factor, epoch)
		trainObj, top1, top2 = train(trainloader, model, critertion, opt, epoch)
		valObj, prec1, prec2 = evaluate(valloader, model, critertion,is_last,args,epoch)
		# stats._update(trainObj, top1, top2, valObj, prec1, prec2)

		filename = []
		if args.store_per_epoch:
			filename.append(os.path.join(args.modeldir, 'net-epoch-%s.pth.tar' % (epoch + 1)))
		else:
			filename.append(os.path.join(args.modeldir, 'checkpoint.pth.tar'))
		filename.append(os.path.join(args.modeldir, 'model_best.pth.tar'))
		save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_prec1,
						 'optimizer': opt.state_dict()}, (prec1 > best_prec1), filename)
		plot_curve(stats, args.modeldir, True)
		sio.savemat(os.path.join(args.modeldir, 'stats.mat'), {'data': stats})

	stats.get_last5()


def train(trainloader, model, criterion, optimizer, epoch):
	batch_time = AvgMeter()
	data_time = AvgMeter()
	losses = AvgMeter()
	top1 = AvgMeter()
	top2 = AvgMeter()
	model.train()
	end = time.time()
	for i, (input, target,true_mask) in enumerate(trainloader):
		data_time.update(time.time() - end)
		input, target, true_mask = input.cuda(), target.cuda().long(),true_mask.cuda()
		mask= model(input)
		# prec1, prec2 = accuracy(out1, target.long(), topk=(1,2))
		# loss =F.binary_cross_entropy_with_logits(mask,true_mask.unsqueeze(1))
		# loss=F.binary_cross_entropy(mask,true_mask.unsqueeze(1))
		loss=criterion(mask,true_mask.unsqueeze(1))
		losses.update(loss.item(), input.size(0))
		# top1.update(prec1[0], input.size(0))
		# top2.update(prec2[0], input.size(0))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		batch_time.update(time.time() - end)
		end = time.time()
		if i % args.print_freq == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
				epoch, i, len(trainloader), batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1, top2=top2))
	return losses.avg, top1.avg, top2.avg


def evaluate(valloader, model, criterion,is_last,args,epoch):
	batch_time = AvgMeter()
	losses = AvgMeter()
	top1 = AvgMeter()
	top2 = AvgMeter()
	model.eval()
	with torch.no_grad():
		end = time.time()
		for i, (input, target,path, true_mask) in enumerate(valloader):
			input, target,true_mask = input.cuda(), target.cuda().long(),true_mask.cuda()
			mask = model(input)
			loss=criterion(mask,true_mask.unsqueeze(1))
			if epoch%5==0:
				plot_mask(input,mask)
			path=path if is_last else None
			losses.update(loss.item(), input.size(0))
			batch_time.update(time.time() - end)

			if i % args.print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
					  'Prec@2 {top1.val:.3f} ({top2.avg:.3f})'.format(
					i, len(valloader), batch_time=batch_time, loss=losses,
					top1=top1, top2=top2))

		print(' * Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f}'.format(top1=top1, top2=top2))

		return losses.avg, top1.avg, top2.avg


class Learning_rate_generater(object):
	"""
	Generate a list of learning rate
	"""

	def __init__(self, method, params, total_epoch):
		if method == 'step':
			lr_factor, lr = self.step(params, total_epoch)
		elif method == 'log':
			lr_factor, lr = self.log(params, total_epoch)
		elif method == 'exp':
			lr_factor, lr = self.exp(params, total_epoch)
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
			lr.append(args.lr * lr_factor[epoch])
		return lr_factor, lr

	def log(self, params, total_epoch):
		min_, max_ = params[:2]
		np_lr = np.logspace(min_, max_, total_epoch)
		lr_factor = []
		lr = []
		for epoch in range(total_epoch):
			lr.append(np_lr[epoch])
			lr_factor.append(np_lr[epoch] / np_lr[0])
		if lr[0] != args.lr:
			args.lr = lr[0]
		return lr_factor, lr


def adjust_learning_rate(optimizer, lr_factor, epoch):
	"""
	:param optimizer:
	:param lr_factor:
	:param epoch:
	:return:
	"""
	print('the lr is set to {0:.5f}'.format(lr_factor[epoch] * args.lr))
	for params_group in optimizer.param_groups:
		params_group['lr'] = lr_factor[epoch] * args.lr

def plot_mask(input,masks):
	mean = np.expand_dims(np.expand_dims([0.275, 0.278, 0.284], axis=1), axis=1)
	std = np.expand_dims(np.expand_dims([0.170, 0.171, 0.173], axis=1), axis=1)
	input = input.cpu().numpy()
	for i in range(input.shape[0]):
		image = np.moveaxis((input[i] * std + mean), 0, -1) * 255
		image=image.astype(np.uint8).copy()
		mask=masks[i]
		mask=mask.cpu().numpy()*255
		mask=np.moveaxis(mask,0,-1).astype(np.uint8())
		mask=cv2.resize(mask,(224,224))
		mask= cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
		image= cv2.addWeighted(image,1.6,mask,1.0,20)
		cv2.imshow('image', image)
		cv2.waitKey(200)
if __name__ == '__main__':
	main()
