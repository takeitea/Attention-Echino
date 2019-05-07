import os
import torch

__all__ = ['restore']


def restore(args, model, optimizer, istrain=True, including_opt=False):
	if os.path.isfile(args.resume) and ('.pth.tar' in args.resume):
		snapthshot = args.resume
	else:
		restore_dir = args.snapshot_dir
		filelist = os.listdir(restore_dir)
		filelist = [x for x in filelist if os.path.isfile(os.path.join(restore_dir, x)) and x.endswith('.pth.tar')]
		if len(filelist) > 0:
			filelist.sort(key=lambda fn: os.path.getmtime(os.path.join(restore_dir, fn)), reverse=True)
			snapthshot = os.path.join(restore_dir, filelist[0])
		else:
			snapthshot = ''
	if os.path.isfile(snapthshot):
		print("++loading checkpoint '{}'".format(snapthshot))
		checkpoint = torch.load(snapthshot)
		try:
			if istrain:
				args.current_epoch = checkpoint['epoch'] + 1
				if including_opt:
					optimizer.load_state_dict(checkpoint['optimizer'])
			old_state_dict=model.state_dict()
			for k,v in checkpoint['state_dict'].items():
				if k in old_state_dict.keys():
					model.load_state_dict(checkpoint['state_dict'])
					return
				elif k[7:] in old_state_dict.keys():
					old_state_dict[k[7:]]=v

			old_state_dict.update()
			model.load_state_dict(old_state_dict)
			print(" loaded checkpoint '{}'(epoch {})".format(snapthshot, checkpoint['epoch']))
			return checkpoint['epoch']
		except KeyError:
				_model_load(model, checkpoint)
		except KeyError:
			print("loading pre-trained values failed")
			raise
	else:
		print("no checkpoint found at {}".format(snapthshot))


def _model_load(model, pretrained_dict):
	model_dict = model.state_dict()
	# if model_dict.keys()[0].startswith('module.'):
	# 	pretrained_dict = {'module.' + k: v for k, v in pretrained_dict.items()}
	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
	print("weights cannot be loaded:")
	print([k for k in model_dict.keys() if k not in pretrained_dict.keys()])
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)