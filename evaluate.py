import argparse
import shutil

import torch
import time
import os
from eval_F1_acc import MY_EVALUATE
from model import resnet18
from utils import AvgMeter, accuracy, plot_curve,restore
from utils import vizNet, Stats, save_checkpoint,loadpartweight
import tqdm
from data import get_data

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
best_prec1 = 0


def arg_pare():
	arg = argparse.ArgumentParser(description=" args of atten-vgg")
	arg.add_argument('-bs', '--batch_size', help='batch size', default=16)
	arg.add_argument('--store_per_epoch', default=False)
	arg.add_argument('--epochs', default=60)
	arg.add_argument('--num_classes', default=9, type=int)
	arg.add_argument('--lr', help='learn rate', default=0.01)
	arg.add_argument('-att', '--attention', help='whether to use attention', default=True)
	arg.add_argument('--img_size', help='the input size', default=224)
	arg.add_argument('--dir', help='the dataset root', default='/data/wen/Dataset/data_maker/classifier/c9/')
	arg.add_argument('--print_freq', default=180, help='the frequency of print infor')
	arg.add_argument('--modeldir', help=' the model viz dir ', default='ResNet18_resize')
	arg.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='# of workers')
	arg.add_argument('--lr_method', help='method of learn rate')
	arg.add_argument('--gpu', default=None, type=str)
	arg.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
	arg.add_argument('--dist_url', default='tcp://127.0.0.01:123', type=str, help='url used to set up')
	arg.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend')
	arg.add_argument('--evaluate', default=True, help='whether to evaluate only')
	arg.add_argument("--resume",default='ResNet18_resize'+'/model_best.pth.tar')
	return arg.parse_args()


args = arg_pare()


def main():
	print('\n loading the dataset ... \n')
	print('\n done \n')
	model=resnet18(pretrained=True,num_classes=9).cuda()
	restore(args,model,optimizer=None,istrain=False)
	trainloader, valloader = get_data(args)

	if args.evaluate:
		evaluate(valloader, model,args.modeldir)
		return
def evaluate(valloader, model,dir):
	batch_time = AvgMeter()
	losses = AvgMeter()
	top1 = AvgMeter()
	top2 = AvgMeter()
	model.eval()
	with torch.no_grad():
		end = time.time()
		for i, (input, target,path) in tqdm.tqdm(enumerate(valloader)):
			input, target = input.cuda(), target.cuda()
			model=model.cuda()
			output= model(input)
			prec1, prec2 = accuracy(output[0], target,dir,path=path, topk=(1, 2))
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
	test='utils/test_list.txt'
	shutil.copy(test,args.modeldir)
	myeval= MY_EVALUATE(args.modeldir)
	myeval.my_acc()
	myeval.my_confusion_matrix()
	myeval.my_classification_report()



if __name__ == '__main__':
	main()