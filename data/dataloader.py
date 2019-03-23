import torch
from .img_preprocess import preprocess_strategy,preprocess_strategy_with_mask
import torchvision.datasets
import torchvision.transforms as transforms
import torch.utils.data
from PIL import Image
from torch.utils.data import distributed
from torchvision.datasets import DatasetFolder
import numpy as np
import os
import re
from torch.utils.data import Dataset
from .img_preprocess import ImgAugTransform
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
	valset = ValFolder(root=args.dir + 'val', transform=val_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32,
											  pin_memory=True)
	valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=32,
											pin_memory=True)
	return trainloader, valloader

def get_data_mask(args, test_path=False):

	input_size = int(250)
	crop_size = int(224)
	tsfm_train = ImgAugTransform(input_size, crop_size)
	func_transforms = []
	func_transforms.append(transforms.Resize(input_size))
	func_transforms.append(transforms.CenterCrop(crop_size))
	func_transforms.append(transforms.ToTensor())
	func_transforms.append(transforms.Normalize(MEANS,STDS))
	tsfm_test = transforms.Compose(func_transforms)

	img_train = MaskFolder(datalist_file='./datafolder/C2_MASK_ROI/train_list.txt', mask=True, transform=tsfm_train)

	img_test = MaskFolder('./datafolder/C2_MASK_ROI/val_list.txt',mask=False, transform=tsfm_test, with_path=True)
	train_loader = torch.utils.data.DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=32)
	val_loader = torch.utils.data.DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=32)
	return train_loader, val_loader

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
			# data = {"image": np.array(sample)}
			sample = self.transform(sample)
		# sample = self.transform(**data)['image']
		if self.target_transform is not None:
			target = self.target_transform(target)
		if self.with_path:
			return sample, target, path
		return sample, target


class ValFolder(MyFolder):
	def __init__(self, root, transform=None, target_transform=None, with_path=True, loader=default_loader):
		self.with_path = with_path
		super(ValFolder, self).__init__(root, transform, target_transform, with_path=True, loader=loader)
		self.imgs = self.samples

	def __getitem__(self, idx):
		path, target = self.samples[idx]
		sample = self.loader(path)
		if self.transform is not None:
			# sample = self.transform(sample)
			# data = {"image": np.array(sample)}
			sample = self.transform(sample)
		# sample = self.transform(**data)['image']
		if self.target_transform is not None:
			target = self.target_transform(target)
		if self.with_path:
			return sample, target, path
		return sample, target


class MaskFolder(Dataset):
	""" Echinococcosis dataset with mask
	"""

	def __init__(self, datalist_file='datafolder/C2_MASK_ROI/train_list.txt',root_dir='/home/wen/PycharmProjects/Attention-Echino',
				 mask=True, transform=None, with_path=False):
		"""

		:param datalist_file: path to txt list
		:param root_dir:  imgdir
		:param transform:
		:param with_path:
		"""
		self.root_dir=root_dir
		self.with_path = with_path
		self.datalist_file = datalist_file
		self.mask = mask
		self.image_list, self.lable_list = self.read_labeled_image_list(self.datalist_file)
		if mask:
			self.mask_list = self.read_mask_list()
		self.transform = transform

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, idx):
		if self.mask:
			image_path = self.image_list[idx]
			mask_name = self.mask_list[idx]
			image = Image.open(image_path).convert('RGB')
			mask = Image.open(mask_name).convert('L')
			if self.transform is not None:
				image, mask = self.transform(image, mask)
			if self.with_path:
				return image, self.lable_list[idx],image_path,mask
			else:
				return image, self.lable_list[idx], mask

		else:
			img_name = self.image_list[idx]
			image = Image.open(img_name).convert('RGB')

			if self.transform is not None:
				image = self.transform(image)
			if self.with_path:
				return img_name, image, self.lable_list[idx]
			else:
				return image, self.lable_list[idx]

	def read_labeled_image_list(self, data_list):

		"""
		Read txt file containing paths to images and ground truth masks
		Args:
			data_dir: path to the directory with images and masks
			data_list: path to the file with lines of the form '/path/to/image
			 /path/to/mask'
		Return :
			Two lists with all file names and masks

	"""
		f = open(data_list, 'r')
		img_name_list = []
		img_labels = []
		for line in f:
			if '.' in line:
				image, labels = line.strip('\n').split(',')
			else:
				if len(line.strip().split()) == 2:
					image, labels = line.strip().split()
					if '.' not in image:
						image += '.jpg'
					labels = int(labels)
				else:
					line = line.strip().split()
					image = line[0]

					labels = map(int, line[1:])

			image = image[len(image.split('/')[0])+1:]
			img_name_list.append(os.path.join(self.root_dir, image))
			img_labels.append(labels)
		return img_name_list, np.array(img_labels, dtype=np.float32)

	def read_mask_list(self, mask_folder='mask'):
		"""
		read the mask list according to image list
		note that the mask image  in different folder  with the same structure of image
		/path/image_folder/train/tp/name.jpg
		/path/mask_folder/train/tp/name.png
		:return: self.mask list
		"""
		mask_lists = []
		datepad = re.compile(r'(.*)/(.*)/(.*/.*/.*)')
		for image in self.image_list:
			m = datepad.match(image)
			mask_list = os.path.join(m.group(1), mask_folder, m.group(3))
			mask_list=mask_list[ :-len(mask_list.split('.')[1])]+'png'
			mask_lists.append(os.path.join(self.root_dir, mask_list))
		return mask_lists
