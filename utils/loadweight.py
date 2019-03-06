"""
load part of the pre-trained parameters
"""
import os
import torch
import torch.utils.model_zoo as model_zoo
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def loadcheckpoint(model, optimizer, args):
	if args.resume:
		if os.path.isfile(args):
			print("load checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']
			best_prec1 = checkpoint['best_prec1']
			model.load_state_dict(checkpoint['state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer'])

			print(" loaded checkpoint '{}'({}) best_prec: {}".format(args.resume, checkpoint['epoch'], best_prec1))

		else:
			print("no checkpoint found at {}".format(args.resume))


def loadpartweight(model):
	old_dict=model.state_dict()
	new_dict=model_zoo.load_url(model_urls['vgg16_bn'])
	count_feat=0
	count_fetch=0
	skip=0
	for k,_ in new_dict.items():
		if 'features' in k:
			count_feat=count_feat+1
	for i in range(count_feat):
		for k in range(i,len(old_dict)):
			if 'num_batches_tracked' in list(old_dict.keys())[k+skip]:
				skip+=1
			if new_dict[list(new_dict.keys())[i]].size()==old_dict[list(old_dict.keys())[k+skip]].size():
				old_dict[list(old_dict.keys())[k+skip]]=list(new_dict.values())[i]
				count_fetch+=1
				break
	old_dict.update()
	model.load_state_dict(old_dict)
	return model

