import argparse
import tqdm
import os
import shutil
import time
from model import EchiNet_18
from pretrainedmodels.models.pnasnet import pnasnet5large
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
from data import get_nature,Pre_Image
from loss import Auxiliary_Loss
from utils import AvgMeter, accuracy, plot_curve, restore
from utils import Stats, save_checkpoint, Learning_rate_generater
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def arg_pare():
	arg = argparse.ArgumentParser(description=" args of train inature")
	arg.add_argument('-bs', '--batch_size', help='batch size', default=40)
	arg.add_argument('--store_per_epoch', default=False)
	arg.add_argument('--epochs', default=30)
	arg.add_argument('--num_classes', default=1010, type=int)
	arg.add_argument('--lr', help='learn rate', default=0.0045)
	arg.add_argument('--img_size', help='the input size', default=331)
	arg.add_argument('--print_freq', default=180, help='the frequency of print infor')
	arg.add_argument('--modeldir', help=' the model viz dir ', default='inature_Echi')
	arg.add_argument('--lr_method', default='step', help='method of learn rate')
	arg.add_argument('--gpu', default=4, type=str)
	arg.add_argument('--test', default=False, help='whether to test only')
	arg.add_argument('--resume',default=False)
	# arg.add_argument('--resume', default='./inature_pnas/model_best.pth.tar', help="whether to load checkpoint")
	arg.add_argument('--start_epoch', default=0)
	arg.add_argument('--op_file_name', default='./result/kaggle_submission.csv')
	arg.add_argument('--demo',default=False)
	return arg.parse_args()


args = arg_pare()


def main():
	best_prec1 = 0
	print('\n loading the dataset ... \n')
	print('\n done \n')
	model =EchiNet_18().cuda()
	opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.90, weight_decay=1e-4)
	if args.resume:
		restore(args, model, opt, istrain=not args.test,including_opt=True)
	if args.demo:
		model=model.eval()
		sr_image=cv2.imread(args.demo)
		image=cv2.resize(sr_image.copy(),(331,331))
		pi=Pre_Image()
		image=pi(image)
		restore(args, model, opt, istrain=not args.test)
		out=model(image)
		score,pred=torch.topk(out,k=1,dim=1)
		pred=pred.cpu().numpy()
		# cat=id2cat(pred)
		cv2.putText(sr_image,str(pred[0][0]),(50,50),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(0,255,255),thickness=1)
		cv2.imwrite('out.jpg',sr_image)
		return

	if args.test :
		restore(args, model, opt, istrain=not args.test)
		model = torch.nn.DataParallel(model, range(args.gpu))
		testloader = get_nature(args.test)
		test(testloader, model, args)
		return
	LR = Learning_rate_generater('step', [16, 22], args.epochs,args)
	print(args)
	model = torch.nn.DataParallel(model, range(args.gpu))
	critertion = torch.nn.CrossEntropyLoss()

	trainloader, valloader = get_nature(args.test)
	if os.path.exists(args.modeldir):
		shutil.rmtree(args.modeldir)
	os.mkdir(args.modeldir)
	stats = Stats(args.modeldir, start_epoch=args.start_epoch, total_epoch=args.epochs)
	for epoch in range(args.epochs):
		is_best = False
		# LR.adjust_learning_rate(opt, LR.lr_factor, epoch)
		LR.cos_anneal_lr(opt, LR.lr, epoch)
		trainObj, top1, top2 = train(trainloader, model, critertion, opt, epoch)
		valObj, prec1, prec2 = evaluate(valloader, model, critertion, args)
		stats._update(trainObj, top1, top2, valObj, prec1, prec2)
		if best_prec1 < prec1:
			best_prec1 = prec1
			is_best = True
		filename = []
		if args.store_per_epoch:
			filename.append(os.path.join(args.modeldir, 'net-epoch-%s.pth.tar' % (epoch + 1)))
		else:
			filename.append(os.path.join(args.modeldir, 'checkpoint.pth.tar'))
		filename.append(os.path.join(args.modeldir, 'model_best.pth.tar'))
		save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_prec1,"global_epoch":
		epoch+1+args.start_epoch, 'optimizer': opt.state_dict()}, is_best, filename)
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
	aloss = Auxiliary_Loss()
	for i, (input, im_id, target, tax_id) in enumerate(trainloader):
		data_time.update(time.time() - end)
		input, target = input.cuda(), target.cuda()
		out1 = model(input)
		loss = criterion(out1, target)+5*aloss(out1,target)
		prec1, prec2 = accuracy(out1, target, topk=(1, 2))
		losses.update(loss.item(), input.size(0))
		top1.update(prec1[0], input.size(0))
		top2.update(prec2[0], input.size(0))

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

def evaluate(valloader, model, criterion, args):
	batch_time = AvgMeter()
	losses = AvgMeter()
	top1 = AvgMeter()
	top2 = AvgMeter()
	model.eval()
	with torch.no_grad():
		end = time.time()
		for i, (input, im_id, target, tax_ids) in enumerate(valloader):
			input = input.cuda()
			target = target.cuda()
			output1 = model(input)
			loss = criterion(output1, target)
			prec1, prec2 = accuracy(output1, target, topk=(1, 2))
			losses.update(loss.item(), input.size(0))
			top1.update(prec1[0], input.size(0))
			top2.update(prec2[0], input.size(0))
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


def test(testloader, model, args):
	batch_time = AvgMeter()
	model.eval()
	if os.path.exists(args.op_file_name):
		os.remove(args.op_file_name)
	with open(args.op_file_name, 'w')as L:
		L.write("id,predicted\n")
	with torch.no_grad():
		end = time.time()
		for i, (input, im_id, target, tax_ids) in tqdm.tqdm(enumerate(testloader)):
			input = input.cuda()
			output1 = model(input)
			np_scores, pred_labels = torch.topk(output1, k=2, dim=1)
			pred_labels = pred_labels.cpu().data.numpy()
			save_preds(im_id.numpy(), np_pred=pred_labels, filename=args.op_file_name)
			batch_time.update(time.time() - end)
			if i % args.print_freq == 0:
				print('Test: [{0}/{1}]\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
					i, len(testloader), batch_time=batch_time))


def save_preds(im_ids, np_pred, filename):
	with open(filename, 'a') as L:
		for ii in range(len(im_ids)):
			L.write(str(im_ids[ii]) + ',' + ' '.join(str(x) for x in np_pred[ii, :]) + '\n')


if __name__ == '__main__':
	main()
