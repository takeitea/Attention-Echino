import argparse
import torch
import time
import os

from model import resnet18
from utils import AvgMeter, accuracy, plot_curve
from utils import get_data, vizNet, Stats, save_checkpoint,loadpartweight
import tqdm

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
	arg.add_argument('--modeldir', help=' the model viz dir ', default='viz_base')
	arg.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='# of workers')
	arg.add_argument('--lr_method', help='method of learn rate')
	arg.add_argument('--gpu', default=None, type=str)
	arg.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
	arg.add_argument('--dist_url', default='tcp://127.0.0.01:123', type=str, help='url used to set up')
	arg.add_argument('--dist_backend', default='gloo', type=str, help='distributed backend')
	arg.add_argument('--evaluate', default=True, help='whether to evaluate only')

	return arg.parse_args()


args = arg_pare()


def main():
	print('\n loading the dataset ... \n')
	print('\n done \n')
	model=resnet18(pretrained=True,num_classes=9,is_val=True).cuda()
	model.load_state_dict(torch.load(os.path.join(args.modeldir,'model_best.pth.tar'))['state_dict'])
	trainloader, valloader = get_data(args)

	# critertion = torch.nn.CrossEntropyLoss(weight=torch.Tensor([5,5,5,1,5,1,1,1,1,])).cuda()
	if args.evaluate:
		evaluate(valloader, model)
		return
def evaluate(valloader, model):
	batch_time = AvgMeter()
	losses = AvgMeter()
	top1 = AvgMeter()
	top2 = AvgMeter()
	model.eval()
	with torch.no_grad():
		end = time.time()
		for i, (input, target) in tqdm.tqdm(enumerate(valloader)):
			input, target = input.cuda(), target.cuda()
			model=model.cuda()
			output= model(input,target)

			prec1, prec2 = accuracy(output, target, path=None, topk=(1, 2))
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



if __name__ == '__main__':
	main()