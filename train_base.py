import argparse
import json
import os
from pytorchcv.model_provider import get_model as ptcv_get_model

import shutil
import time
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_data
from loss import Auxiliary_Loss,ComEnLoss
from model import resnet18,EchiNet_18,drn_c_26,shufflenetv2,drn_a_50
from utils import AvgMeter, accuracy, plot_curve, restore
from utils import Stats, save_checkpoint,Learning_rate_generater

os.environ["CUDA_VISIBLE_DEVICES"] = "5,2,3,4"



def arg_pare():
	arg = argparse.ArgumentParser(description=" args of base")
	arg.add_argument('-bs', '--batch_size', help='batch size', default=40)
	arg.add_argument('--store_per_epoch', default=False)
	arg.add_argument('--epochs', default=25)
	arg.add_argument('--num_classes', default=9, type=int)
	arg.add_argument('--lr', help='learn rate', default=0.001)
	arg.add_argument('--img_size', help='the input size', default=224)
	arg.add_argument('--dir',help="",default='./datafolder/c9_224/')
	# arg.add_argument('--dir',help="",default= './datafolder/grade/CLCE1/')
	# arg.add_argument('--dir', help='the dataset root',default= '/home/whx/dataset_maker/VOC_maker/detect_roi/roi_padding_224/CE234/')
	arg.add_argument('--print_freq', default=180, help='the frequency of print infor')
	arg.add_argument('--aug',help="whether to augment ",default=True)
	arg.add_argument('--modeldir', help=' the model viz dir ', default='drk53_aug')
	arg.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='# of workers')
	arg.add_argument('--lr_method',default='step',help='method of learn rate')
	arg.add_argument('--gpu', default=4, type=str)
	arg.add_argument('--evaluate', default=False, help='whether to evaluate only')
	arg.add_argument('--resume',default=False)
	# arg.add_argument('--resume', default='./aug_18_224_CE2CE3CE4/model_best.pth.tar', help="whether to load checkpoint")
	arg.add_argument('--start_epoch', default=0)
	return arg.parse_args()


args = arg_pare()

def main():
	best_prec1 = 0
	print('\n loading the dataset ... \n')
	print('\n done \n')
	# model=(pretrained=True)
	model=ptcv_get_model( "darknet53",pretrained=False)
	model.load_state_dict(torch.load("./model/darknet53.pth"))
	# model.load_state_dict(torch.load('./model/shufflenetv2_x1.pth.tar'))
	# model.classifier=  nn.Sequential(nn.Linear(model.stage_out_channels[-1], args.num_classes))
	model.fc=nn.Linear(2048,9)
	model.cuda()
	LR = Learning_rate_generater('step', [15,20], args.epochs,args)

	opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.90, weight_decay=1e-4)
	# cmopt=optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=1e-4)
	print(args)
	# vizNet(model, args.modeldir)
	if args.evaluate:
		restore(args, model, opt,  istrain=not args.evaluate)
	model = torch.nn.DataParallel(model, range(args.gpu))
	trainloader, valloader,test_loader = get_data(args)
	critertion = torch.nn.CrossEntropyLoss()
	if args.test:
		evaluate(test_loader,model,critertion,args,True)
	if args.evaluate:
		evaluate(valloader, model, critertion,args,True)
		return
	if os.path.exists(args.modeldir):
		shutil.rmtree(args.modeldir)
	os.mkdir(args.modeldir)
	stats = Stats(args.modeldir, start_epoch=args.start_epoch, total_epoch=args.epochs)
	for epoch in range(args.epochs):
		is_best=False
		is_last=epoch==args.epochs-1
		# adjust_learning_rate(opt, LR.lr_factor, epoch)
		LR.cos_anneal_lr(opt,LR.lr,epoch)
		trainObj, top1, top2 = train(trainloader, model, critertion, opt, epoch)
		valObj, prec1, prec2 = evaluate(valloader, model, critertion,args,is_last)

		stats._update(trainObj, top1, top2, valObj, prec1, prec2)
		if best_prec1<prec1:
			best_prec1=prec1
			is_best=True
		filename = []
		if args.store_per_epoch:
			filename.append(os.path.join(args.modeldir, 'net-epoch-%s.pth.tar' % (epoch + 1)))
		else:
			filename.append(os.path.join(args.modeldir, 'checkpoint.pth.tar'))
		filename.append(os.path.join(args.modeldir, 'model_best.pth.tar'))
		save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_prec1,
						 'optimizer': opt.state_dict()}, is_best, filename)
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
	aloss=Auxiliary_Loss(k=2)
	# cmloss=ComEnLoss()

	for i, (input, target) in enumerate(trainloader):
		data_time.update(time.time() - end)
		input, target = input.cuda(), target.cuda()
		out1,_= model(input)

		loss = criterion(out1, target)+2*aloss(out1,target)
		prec1, prec2 = accuracy(out1, target,topk=(1, 2))
		losses.update(loss.item(), input.size(0))
		top1.update(prec1[0], input.size(0))
		top2.update(prec2[0], input.size(0))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		batch_time.update(time.time() - end)
		end = time.time()
		# out1=model(input)
		# loss=cmloss(out1,target)
		# cmopt.zero_grad()
		# loss.backward()
		# cmopt.step()
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


def evaluate(valloader, model, criterion,args,is_last):
	id2class = {id: class_ for class_, id in valloader.dataset.class_to_idx.items()}
	batch_time = AvgMeter()
	losses = AvgMeter()
	top1 = AvgMeter()
	top2 = AvgMeter()
	model.eval()
	data = {}
	data['path'] = []
	data['feature'] = []
	data['label'] = []
	with torch.no_grad():
		end = time.time()
		for i, (input, target,path) in enumerate(valloader):
			input = input.cuda()
			target=target.cuda()
			output1,_= model(input)
			loss = criterion(output1, target)
			prec1, prec2 = accuracy(output1, target, topk=(1, 2))
			if is_last:
				# np_last_featmaps=model.get_localization_maps().cpu().data.numpy()
				np_scores, pred_labels = torch.topk(output1, k=2, dim=1)
				pred_labels=pred_labels.cpu().data.numpy()
				np_scores=np_scores.cpu().data.numpy()
				target=target.cpu()
				# save_atten.get_masked_img(path,np_last_featmaps,pred_labels,target.cpu().numpy())
				data['path'].extend(path)
				data['feature'].extend(output1.cpu().numpy().tolist())
				data['label'].extend([int(i) for i in target.numpy()])
				# model.saved_erased_img(img_path=path)

				save_txt(np_scores,pred_labels,path,id2class)
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
		with open('data.json', 'w') as out:
			json.dump(data, out)
		return losses.avg, top1.avg, top2.avg



def save_txt(np_score,np_pred,path,id2class):
	with open('CE2CE3CE4.txt','a') as L:
		for path_,pred,score in zip(path,np_pred,np_score):
			L.write((path_.split('/')[-1]+' '+id2class[pred[0]]+' '+id2class[pred[1]]+' '+str(score[0])+' '+str(score[1])+'\n'))

if __name__ == '__main__':
	main()
