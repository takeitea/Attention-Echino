import torch
from .imgaug import preprocess_strategy
import torchvision.datasets
import torchvision.transforms as trans
import torch.utils.data
from PIL import Image
from torch.utils.data import distributed
from torchvision.datasets import DatasetFolder
import numpy as np

MEANS = [0.275, 0.278, 0.284]
STDS = [0.170, 0.171, 0.173]

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')


def accimage_loader(path):
	import accimage
	try:
		return accimage.Image(path)
	except IOError:
		# Potentially a decoding problem, fall back to PIL.Image
		return pil_loader(path)


def default_loader(path):
	from torchvision import get_image_backend
	if get_image_backend() == 'accimage':
		return accimage_loader(path)
	else:
		return pil_loader(path)


def get_data(args):
	trans_train, val_train = preprocess_strategy()
	trainset = MyFolder(root=args.dir + 'train', transform=trans_train)
	valset = MyFolder(root=args.dir + 'val', transform=val_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32,
											  pin_memory=True)
	valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=32,
											pin_memory=True)
	return trainloader, valloader


class MyFolder(DatasetFolder):
	"""A generic data loader where the images are arranged in this way: ::

		root/dog/xxx.png
		root/dog/xxy.png
		root/dog/xxz.png

		root/cat/123.png
		root/cat/nsdf3.png
		root/cat/asd932_.png

	Args:
		root (string): Root directory path.
		transform (callable, optional): A function/transform that  takes in an PIL image
			and returns a transformed version. E.g, ``transforms.RandomCrop``
		target_transform (callable, optional): A function/transform that takes in the
			target and transforms it.
		loader (callable, optional): A function to load an image given its path.

	 Attributes:
		classes (list): List of the class names.
		class_to_idx (dict): Dict with items (class_name, class_index).
		imgs (list): List of (image path, class_index) tuples
	"""

	def __init__(self, root, transform=None, target_transform=None, with_path=None,
				 loader=default_loader):
		self.with_path = with_path
		super(MyFolder, self).__init__(root, loader, IMG_EXTENSIONS,
									   transform=transform,
									   target_transform=target_transform)
		self.imgs = self.samples

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (sample, target) where target is class_index of the target class.
		"""
		path, target = self.samples[index]
		sample = self.loader(path)
		if self.transform is not None:
			data = {"image": np.array(sample)}
			# sample=self.transform(sample)
			sample = self.transform(**data)['image']
		if self.target_transform is not None:
			target = self.target_transform(target)
		if self.with_path:
			return sample, target, path
		return sample, target


class ValFolder(MyFolder):
	def __init__(self, root, transform=None, target_transform=None, with_path=None, loader=default_loader):
		self.with_path = with_path
		super(ValFolder, self).__init__(root, transform, target_transform, with_path=True, loader=loader)
		self.imgs = self.samples

	def __getitem__(self, idx):
		path, target = self.samples[idx]
		sample = self.loader(path)
		if self.transform is not None:
			sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)
		if self.with_path:
			return sample, target, path
		return sample, target
