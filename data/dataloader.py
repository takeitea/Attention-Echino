import torch
import cv2
import json
from .img_preprocess import preprocess_strategy,ImgAugTransform
import torchvision.datasets
from .prepare_image import ResizePadding
import torchvision.transforms as transforms
import torch.utils.data
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.datasets import DatasetFolder
import numpy as np
import os
import re
from torch.utils.data import Dataset

MEANS = [0.275, 0.278, 0.284]
STDS = [0.170, 0.171, 0.173]

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


# id2cat=

def pil_loader(path):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		return img.convert('RGB')
def cv2_loader(path):
	return cv2.imread(path)

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
	trainset = MyFolder(root=args.dir + 'train', transform=trans_train,aug=args.aug)
	valset = ValFolder(root=args.dir + 'val', transform=val_train,aug=args.aug)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32,
											  pin_memory=True)
	valloader = torch.utils.data.DataLoader(valset, batch_size=50, shuffle=False, num_workers=32,
											pin_memory=True)
	return trainloader, valloader


class Detectfolder(Dataset):
	def __init__(self, is_train=True, transfer=None,aug=False,test=False,outsample=False,big=False,sub=''):
		self.root = '/data/wen/data/C9/'
		self.test=test
		self.outsample=outsample
		pre='aug' if aug else ''
		pre='big'+pre if big else pre
		# assert not (is_train  and test)
		# assert not (aug and big)
		if is_train:
			self.anno_path=self.root+'train' +pre+'.txt'
			self.image_path=self.root+'train'+pre
			if sub:
				self.anno_path = self.root+pre+'/sub'+sub+'.txt'
				if aug:
					self.image_path = self.root +pre+'sub' +sub
		elif test:

			self.anno_path = self.root + 'output.txt'
			if sub:
				self.anno_path=self.root+'test_sub'+sub+'.txt'
			self.image_path = self.root + 'test'
		else:
			if sub:
				self.anno_path=self.root+'test_sub'+sub+'.txt'
			else:
				self.anno_path=self.root+'test.txt'
			self.image_path = self.root + 'test'

		self.name2id = {'AE1': 0, 'AE2': 1, 'AE3': 2, 'CE1': 3, 'CE2': 4, 'CE3': 5, 'CE4': 6, 'CE5': 7, 'CL': 8}
		if sub:
			subclass = [['CL', 'CE1'], ['CE2', 'CE3', 'CE4'], ['AE1', 'AE2', 'AE3']]
			self.name2id={ subclass[int(sub)][i]:i for i in range(len(subclass[int(sub)]))}
		self.id2name = {self.name2id[k]: k for k in self.name2id.keys()}
		self.samples = []
		self.image_size = [224, 224]
		self.trans = transfer
		self.small=ResizePadding(size=200)
		self.middle=ResizePadding(size=250)
		self.large=ResizePadding(size=300)
		for line in open(self.anno_path, 'r').readlines():
			if aug and is_train:
				sample=line.strip().split(' ')[:]
				del sample[1]
			elif test:
				sample=line.strip().split(' ')[:-1]
			else:
				sample = line.strip().split(' ')[:]
			self.samples.append(sample)

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		sample = self.samples[idx]
		name = sample[0]
		image_path = os.path.join(self.image_path, name)

		image = cv2.imread(image_path)

		image = np.array(image, dtype=np.uint8)
		image_size = image.shape
		x1, y1, x2, y2 = map(int,sample[2:])

		full_image = image[y1:y2, x1:x2,:].copy()
		zoomin_image=image[int(y1+0.1*(y2-y1)):int(y2-0.1*(y2-y1)),int(x1+0.1*(x2-x1)):int(x2-0.1*(x2-x1)),:].copy()
		zoomout_image=image[max(0,int(y1-0.1*(y2-y1))):min(image_size[0],int(y2+0.1*(y2-y1))),max(0,int(x1-0.1*(x2-x1))):
		 			  min(image_size[1],int(x2+0.1*(x2-x1))),:].copy()
		small_image=self.small.transform(full_image.copy())
		middle_image=self.middle.transform(full_image.copy())
		large_image=self.large.transform(full_image.copy())
		target = self.name2id[sample[1]]

		if self.trans:
			small_image=self.trans(Image.fromarray(small_image))
			middle_image=self.trans(Image.fromarray(middle_image))
			large_image=self.trans(Image.fromarray(large_image))
			full_image=self.trans(Image.fromarray(full_image))
			zoomout_image=self.trans(Image.fromarray(zoomout_image))
			zoomin_image=self.trans(Image.fromarray(zoomin_image))

		inputs=torch.stack(( zoomin_image,zoomout_image,full_image,small_image,middle_image ,large_image),dim=0)
		if self.outsample:
			return inputs,target,sample
		return inputs, target

def get_RE_data(args):
	trans_train, val_train = preprocess_strategy()
	trainset = Detectfolder(is_train=True, transfer=trans_train,sub=args.sub)
	valset = Detectfolder(is_train=False, transfer=val_train,outsample=True,sub=args.sub)
	test=Detectfolder(is_train=False,transfer=val_train,test=args.test,sub=args.sub,outsample=True)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32,
											  pin_memory=True)
	valloader = torch.utils.data.DataLoader(valset, batch_size=50, shuffle=False, num_workers=32,
											pin_memory=True)
	testloader=torch.utils.data.DataLoader(test,batch_size=50,shuffle=False,num_workers=32,pin_memory=True)
	return trainloader, valloader,testloader




def get_nature(is_test):
	root_base = '/data/wen/data/fgvc9/'
	train_file = root_base + 'input/train2019.json'
	val_file = root_base + 'input/val2019.json'
	test_file = root_base + 'input/test2019.json'
	root = root_base + 'input/train_val2019/'
	if is_test:
		test_dataset = INAT(root=root_base + 'input/test2019/', ann_file=test_file, is_train=False)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=40, shuffle=False, num_workers=32,
												  pin_memory=True)
		return test_loader

	train_dataset = INAT(root=root, ann_file=train_file, is_train=True)
	val_dataset = INAT(root=root, ann_file=val_file, is_train=False)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True,
											   num_workers=32, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False, num_workers=32,
											 pin_memory=True)

	return train_loader, val_loader


def get_with_mask(args):
	"""
	get the masked images
		old_dict = model.state_dict()
		new_dict = model_zoo.load_url(model_urls['resnet18'])
		for k, v in new_dict.items():
			if k in old_dict.keys() and old_dict[k].size() == new_dict[k].size():
				old_dict[k] = v
		old_dict.update()
		model.load_state_dict(old_dict, strict=False):param args:
	:return:
	"""
	trans_train, val_train = preprocess_strategy()
	trainset = MaskedFolder(root=args.dir + 'train', transform=trans_train)
	valset = MaskedFolder(root=args.dir + 'val', transform=val_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=32,
											  pin_memory=True)
	valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=32,
											pin_memory=True)
	return trainloader, valloader


def get_data_mask(args, test_path=False):
	input_size = int(224)
	crop_size = int(224)
	tsfm_train = ImgAugTransform(input_size, crop_size)
	func_transforms = []
	func_transforms.append(transforms.Resize(input_size))
	func_transforms.append(transforms.CenterCrop(crop_size))
	func_transforms.append(transforms.ToTensor())
	func_transforms.append(transforms.Normalize(MEANS, STDS))
	# tsfm_test = transforms.Compose(func_transforms)
	tsfm_test = ImgAugTransform(input_size, crop_size)
	img_train = MaskFolder(datalist_file='./datafolder/c2_mask/train_list.txt', mask=True, transform=tsfm_train)

	img_test = MaskFolder('./datafolder/c2_mask/val_list.txt', mask=True, transform=tsfm_test, with_path=True)
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
				 loader=default_loader,aug=False):
		self.with_path = with_path
		super(MyFolder, self).__init__(root, loader, IMG_EXTENSIONS,
									   transform=transform,
									   target_transform=target_transform)
		self.aug=aug
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
			if self.aug:
				data = {"image": np.array(sample)}
				sample = self.transform(**data)['image']
			else:
				sample = self.transform(sample)

		if self.target_transform is not None:
			target = self.target_transform(target)
		if self.with_path:
			return sample, target, path
		return sample, target


class MaskedFolder(DatasetFolder):
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

	def __init__(self, root, transform=None, target_transform=None, with_path=None, aug=False,
				 loader=default_loader):
		self.with_path = with_path
		self.root = root
		self.aug = aug
		self.datpad = re.compile(r'(.*)/(.*)/(.*/.*/.*)(\.jpg$)')
		super(MaskedFolder, self).__init__(root, loader, IMG_EXTENSIONS,
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

		m = self.datpad.match(path)
		mask1_path = os.path.join(m.group(1), 'mask', m.group(3) + '.png')
		mask2_path = os.path.join(m.group(1), 'mask_hollow', m.group(3) + '.png')

		sample = self.loader(path)
		mask1 = self.loader(mask1_path)
		mask2 = self.loader(mask2_path)
		sample = np.array(sample)
		mask1 = np.array(mask1)
		mask2 = np.array(mask2)
		mask1[mask1 > 0] = 1
		mask2[mask2 > 0] = 1
		print(m.group(3))
		# if sample.shape != mask1.shape or sample.shape!=mask2.shape:
		# 	print(m.group(3))
		sample[..., 1] *= mask1[..., 1]
		sample[..., 2] *= mask2[..., 2]
		plt.imshow(sample)
		plt.show()
		sample = Image.fromarray(sample)

		if self.transform is not None:
			if self.aug:
				data = {"image": np.array(sample)}

				sample = self.transform(**data)['image']
			else:
				sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)
		if self.with_path:
			return sample, target, path
		return sample, target


class ValFolder(MyFolder):
	def __init__(self, root, transform=None, target_transform=None, with_path=True, loader=default_loader, aug=False):
		self.with_path = with_path
		super(ValFolder, self).__init__(root, transform, target_transform, with_path=True, loader=loader)
		self.imgs = self.samples
		self.aug = aug

	def __getitem__(self, idx):
		path, target = self.samples[idx]
		sample = self.loader(path)
		if self.transform is not None:
			if self.aug:

				data = {"image": np.array(sample)}

				sample = self.transform(**data)['image']
			else:
				sample = self.transform(sample)
		if self.target_transform is not None:
			target = self.target_transform(target)
		if self.with_path:
			return sample, target, path
		return sample, target





class MaskFolder(Dataset):
	""" Echinococcosis dataset with mask
	"""

	def __init__(self, datalist_file='datafolder/c2_mask/train_list.txt',
				 root_dir='/home/wen/PycharmProjects/Attention-Echino',
				 mask=True, transform=None, with_path=False):
		"""

		:param datalist_file: path to txt list
		:param root_dir:  imgdir
		:param transform:
		:param with_path:
		"""
		self.root_dir = root_dir
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
				return image, self.lable_list[idx], image_path, mask
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

			image = image[len(image.split('/')[0]) + 1:]
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
			mask_list = mask_list[:-len(mask_list.split('.')[1])] + 'png'
			mask_lists.append(os.path.join(self.root_dir, mask_list))
		return mask_lists


class INAT(Dataset):
	def __init__(self, root, ann_file, is_train=True):
		print('load annotations from :' + os.path.basename(ann_file))
		with open(ann_file) as data_file:
			ann_data = json.load(data_file)
		self.imgs = [ann['file_name'] for ann in ann_data['images']]
		self.ids = [ann['id'] for ann in ann_data['images']]

		if 'annotations' in ann_data.keys():
			self.classes = [aa['category_id'] for aa in ann_data['annotations']]
		else:
			self.classes = [0] * len(self.imgs)
		self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
		self.taxonomy, self.classes_taxonomic = self.load_taxonomy(ann_data, self.tax_levels, self.classes)
		print('\t ' + str(len(self.imgs)) + 'images')
		print('\t ' + str(len(set(self.classes))) + 'classes')
		self.root = root
		self.is_train = is_train
		self.loader = default_loader

		self.im_size = [224, 224]
		self.mu_data = [0.485, 0.456, 0.406]
		self.std_data = [0.229, 0.224, 0.225]
		self.brightness = 0.4
		self.contrast = 0.4
		self.saturation = 0.4
		self.hue = 0.25
		self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
		self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
		self.flip_aug = transforms.RandomHorizontalFlip()
		self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)
		self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
		self.tensor_aug = transforms.ToTensor()

	def __getitem__(self, index):
		path = self.root + self.imgs[index]
		im_id = self.ids[index]
		img = self.loader(path)
		species_id = self.classes[index]
		tax_ids = self.classes_taxonomic[species_id]

		if self.is_train:
			img = self.scale_aug(img)
			img = self.flip_aug(img)
			img = self.color_aug(img)
		else:
			img = self.center_crop(img)
		img = self.tensor_aug(img)
		img = self.norm_aug(img)
		return img, im_id, species_id, tax_ids

	def __len__(self):
		return len(self.imgs)

	def load_taxonomy(self, ann_data, tax_levels, classes):
		taxonomy = {}
		if 'categories' in ann_data.keys():
			num_classes = len(ann_data['categories'])
			for tt in tax_levels:
				tax_data = [aa[tt] for aa in ann_data['categories']]
				_, tax_id = np.unique(tax_data, return_inverse=True)
				taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
		else:
			for tt in tax_levels:
				taxonomy[tt] = dict(zip([0], [0]))
		classes_taxonomic = {}
		for cc in np.unique(classes):
			tax_ids = [0] * len(tax_levels)
			for ii, tt in enumerate(tax_levels):
				tax_ids[ii] = taxonomy[tt][cc]
			classes_taxonomic[cc] = tax_ids
		return taxonomy, classes_taxonomic


class Pre_Image:
	def __init__(self):
		self.im_size = [331, 331]
		self.mu_data = [0.485, 0.456, 0.406]
		self.std_data = [0.229, 0.224, 0.225]
		self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
		self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)
		self.tensor_aug = transforms.ToTensor()

	def __call__(self, image):
		# image=self.center_crop(image)
		image = self.tensor_aug(image)
		image = self.norm_aug(image)
		image = image.cuda()
		return image.unsqueeze(0)
