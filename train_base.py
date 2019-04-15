import argparse
import json
import os
import shutil
import time
import scipy.io as sio
import torch
import torch.optim as optim

from data import get_data
from loss import Auxiliary_Loss,IMAE
from model import drn_c_26
from utils import AvgMeter, accuracy, plot_curve, restore
from utils import SAVE_ATTEN
from utils import Stats, save_checkpoint,Learning_rate_generater

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"



def arg_pare():
	arg = argparse.ArgumentParser(description=" args of base")
	arg.add_argument('-bs', '--batch_size', help='batch size', default=40)
	arg.add_argument('--store_per_epoch', default=False)
	arg.add_argument('--epochs', default=30)
	arg.add_argument('--num_classes', default=9, type=int)
	arg.add_argument('--lr', help='learn rate', default=0.001)
	arg.add_argument('-att', '--attention', help='whether to use attention', default=True)
	arg.add_argument('--img_size', help='the input size', default=224)
	arg.add_argument('--dir', help='the dataset root', default='./datafolder/c9/')
	arg.add_argument('--print_freq', default=180, help='the frequency of print infor')
	arg.add_argument('--modeldir', help=' the model viz dir ', default='drn_imae')
	arg.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='# of workers')
	arg.add_argument('--lr_method',default='step',help='method of learn rate')
	arg.add_argument('--gpu', default=4, type=str)
	arg.add_argument('--evaluate', default=False, help='whether to evaluate only')
	arg.add_argument('--resume', default='./result/drc_aloss/model_best.pth.tar', help="whether to load checkpoint")
	arg.add_argument('--start_epoch', default=0)
	return arg.parse_args()


args = arg_pare()


def main():
	best_prec1 = 0
	print('\n loading the dataset ... \n')
	print('\n done \n')
	model = drn_c_26(pretrained=True).cuda()
	model.fc=torch.nn.Linear(512,9).cuda()

	model.cuda()
	LR = Learning_rate_generater('step', [17,25], args.epochs,args)
	# LR.plot_lr()
	opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.90, weight_decay=1e-4)
	print(args)
	# vizNet(model, args.modeldir)
	if args.evaluate:
		restore(args, model, opt,  istrain=not args.evaluate)
	model = torch.nn.DataParallel(model, range(args.gpu))
	trainloader, valloader = get_data(args)
	# critertion = torch.nn.CrossEntropyLoss()
	critertion=IMAE(8)
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
	aloss=Auxiliary_Loss()
	for i, (input, target) in enumerate(trainloader):
		data_time.update(time.time() - end)
		input, target = input.cuda(), target.cuda()
		out1= model(input)

		loss = criterion(out1, target)+5*aloss(out1,target)
		prec1, prec2 = accuracy(out1, target,topk=(1, 2))
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


def evaluate(valloader, model, criterion,args,is_last):
	batch_time = AvgMeter()
	losses = AvgMeter()
	top1 = AvgMeter()
	top2 = AvgMeter()
	model.eval()
	data = {}
	data['path'] = []
	data['feature'] = []
	data['label'] = []
	save_atten=SAVE_ATTEN('./save_c9',dataset='c9')
	with torch.no_grad():
		end = time.time()
		for i, (input, target,path) in enumerate(valloader):
			input = input.cuda()
			target=target.cuda()
			output1= model(input)


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
				save_txt(np_scores,pred_labels,path)
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



def save_txt(np_score,np_pred,path):
	with open('result.txt','a') as L:
		L.write(str(path)+str(np_score[0])+ str(np_score[1])+str(np_pred[0])+str(np_pred[1])+'\n')


if __name__ == '__main__':
	main()
