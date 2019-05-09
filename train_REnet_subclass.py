import argparse
import torch
import time
import numpy as np
import torch.optim as optim
import os
import scipy.io as sio
from model import REnet

from utils import AvgMeter, accuracy, plot_curve,accuracy_lstm,restore
from utils import vizNet, Stats, save_checkpoint, loadpartweight,Learning_rate_generater
from loss import list_loss, ranking_loss, MultiLoss,KL_Loss,Auxiliary_Loss_Seq_CC,IMAE
from data import get_RE_data
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

def arg_pare():
	arg = argparse.ArgumentParser(description=" args of resnet18")
	arg.add_argument('-bs', '--batch_size', help='batch size', default=42)
	arg.add_argument('--store_per_epoch', default=False)
	arg.add_argument('--epochs', default=12)
	arg.add_argument('--num_classes', default=9, type=int)
	arg.add_argument('--lr', help='learn rate', default=0.001)
	arg.add_argument('--print_freq', default=180, help='the frequency of print information')
	arg.add_argument('--modeldir', help=' the model viz dir ', default='train_darkn_lstm_padding')
	arg.add_argument('--lr_method', help='method of learn rate')
	arg.add_argument('--gpu', default=3, type=str)
	arg.add_argument('--test',default=False)
	arg.add_argument('--evaluate', default=False, help='whether to evaluate only')
	# arg.add_argument('--resume',default='./train_darkn_lstm_padding_best/model_best.pth.tar')
	arg.add_argument('--resume', default=False, help="whether to load checkpoint")
	arg.add_argument('--start_epoch', default=0)
	arg.add_argument('--sub',help='subclass num',default='0')
	arg.add_argument('--weight_decay', default=1e-4)
	return arg.parse_args()


args = arg_pare()
subclass=[['CL','CE1'],['CE2','CE3','CE4'],['AE1','AE2','AE3']]
def main():
	print('\n loading the dataset ... \n')
	print('\n done \n')
	model = REnet().cuda()
	# model.pretrained_model.load_state_dict(torch.load('./result/ResNet18_aloss/model_best.pth.tar')["state_dict"])
	LR = Learning_rate_generater('step', [5,9 ], args.epochs,args)
	# LR.plot_lr()

	opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.90, weight_decay=args.weight_decay)
	# opt=optim.Adam(model.parameters(),lr=args.lr,weight_decay=1e-4)
	if args.resume:
		restore(args, model, opt, istrain=not args.evaluate)
	print(args)
	# plot network
	# vizNet(model, args.modeldir)
	model = torch.nn.DataParallel(model, range(args.gpu))
	trainloader, valloader,testloader = get_RE_data(args)
	multiloss=MultiLoss().cuda()
	critertion = torch.nn.CrossEntropyLoss().cuda()
	if args.test:
		test(testloader,model,args=args)
		return
	if args.evaluate:
		evaluate(valloader, model, critertion,is_last=True,args=args,epoch=args.epochs)
		return
	if not os.path.exists(args.modeldir):
		os.mkdir(args.modeldir)
	stats = Stats(args.modeldir, start_epoch=0,total_epoch=args.epochs)
	for epoch in range(args.epochs):
		is_last=epoch==args.epochs-1
		LR.cos_anneal_lr(opt,LR.lr,epoch)
		trainObj, top1, top2 = train(trainloader, model, critertion, opt, epoch, multiloss)
		valObj, prec1, prec2 = evaluate(valloader, model,  critertion,epoch,is_last,args,multiloss)
		stats._update(trainObj, top1, top2, valObj, prec1, prec2)
		filename = []
		if args.store_per_epoch:
			filename.append(os.path.join(args.modeldir, 'net-epoch-%s.pth.tar' % (epoch + 1)))
		else:
			filename.append(os.path.join(args.modeldir, 'checkpoint.pth.tar'))
		filename.append(os.path.join(args.modeldir, 'model_best.pth.tar'))
		save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(),
						 'optimizer': opt.state_dict()}, (True), filename)

		plot_curve(stats, args.modeldir, True)
		sio.savemat(os.path.join(args.modeldir, 'stats.mat'), {'data': stats})

	stats.get_lastn(3)


def train(trainloader, model, criterion, optimizer, epoch, multiloss):
	batch_time = AvgMeter()
	data_time = AvgMeter()
	losses = AvgMeter()
	top1 = AvgMeter()
	top2 = AvgMeter()
	model.train()
	end = time.time()
	aloss=Auxiliary_Loss_Seq_CC(k=2)

	for i, (input, target) in enumerate(trainloader):
		data_time.update(time.time() - end)
		optimizer.zero_grad()
		input, target = input.cuda(), target.cuda()
		lstm_out = model(input)
		loss=multiloss(lstm_out,target)
		loss.backward()
		optimizer.step()
		prec1, prec2 = accuracy_lstm(lstm_out, target,args.modeldir, path=None, topk=(1, 2))
		losses.update(loss.item(), input.size(0))
		top1.update(prec1[0], input.size(0))
		top2.update(prec2[0], input.size(0))
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


def evaluate(valloader, model, criterion,epoch,is_last,args,multiloss):
	batch_time = AvgMeter()
	losses = AvgMeter()
	top1 = AvgMeter()
	top2 = AvgMeter()
	model.eval()
	with torch.no_grad():
		end = time.time()
		for i, (input, target,_) in enumerate(valloader):

			input, target = input.cuda(), target.cuda()
			all_logits= model(input)
			# loss=criterion(all_logits,target)

			loss = multiloss(all_logits, target)
			path =  None
			prec1, prec2 = accuracy_lstm(all_logits, target,  args.modeldir,path=path,topk=(1, 2))

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

def test(testloader, model,args):
	id2class = testloader.dataset.id2name
	model.eval()
	with torch.no_grad():
		for i, (input, target,samples) in enumerate(testloader):
			input, target = input.cuda(), target.cuda()
			output= model(input)
			output =output.view(output.size(0), output.size(1), -1)
			output = torch.mean(output, dim=1)
			_, preds = output.topk(1, 1)
			preds=preds.cpu().numpy()
			save_txt(samples,preds,id2class)
def save_txt(samples,preds,id2class):
	with open('out_re.txt','a') as L:
		for pred,idx in zip(preds,np.arange(len(preds))):
			L.write(' '.join([samples[0][idx],id2class[pred[0]],samples[2][idx],samples[3][idx],samples[4][idx],samples[5][idx]])+'\n')

if __name__ == '__main__':
	main()