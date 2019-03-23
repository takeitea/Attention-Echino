from torch.utils.data import Dataset
import numpy as np
import os
import re
from PIL import Image


class DATASET(Dataset):

	""" Echinococcosis dataset
	"""
	def __init__(self,datalist_file,root_dir,mask=None ,transform=None,with_path=False):
		"""

		:param datalist_file: path to txt list
		:param root_dir:  imgdir
		:param transform:
		:param with_path:
		"""
		self.root_dir=root_dir
		self.with_path=with_path
		self.datalist_file=datalist_file
		self.mask=mask
		self.image_list,self.lable_list=self.read_labeled_image_list( self.root_dir,self.datalist_file )
		if mask:
			self.mask_list=self.read_mask_list()
		self.transform=transform

	def __len__(self):
		return len(self.image_list)
	def __getitem__(self,idx):
		if self.mask:
			img_name=self.image_list[idx]
			mask_name=self.mask_list[idx]
			if self.root_dir>'':
				img_name=os.path.join(self.root_dir,img_name)
				mask_name=os.path.join(self.root_dir,mask_name)
			image=Image.open(img_name).convert('RGB')
			mask=Image.open(mask_name).convert('L')
			if self.transform is not None:
				image,mask1=self.transform(image,mask)
			if self.with_path:
				return img_name,image,self.lable_list[idx],mask1
			else:
				return image,self.lable_list[idx],mask1

		else:
			img_name=self.image_list[idx]
			if self.root_dir>'':
				img_name=os.path.join(self.root_dir,img_name)
			image=Image.open(img_name).convert('RGB')

			if self.transform is not None:
				image=self.transform(image)
			if self.with_path:
				return img_name,image,self.lable_list[idx]
			else:
				return image,self.lable_list[idx]
	def read_labeled_image_list(self,data_dir,data_list):

		"""
		Read txt file containing paths to images and ground truth masks
		Args:
			data_dir: path to the directory with images and masks
			data_list: path to the file with lines of the form '/path/to/image
			 /path/to/mask'
		Return :
			Two lists with all file names and masks

	"""
		f=open(data_list,'r')
		img_name_list=[]
		img_labels=[]
		for line in f:
			if '.' in line:
				image,labels=line.strip('\n').split(',')
			else:
				if len(line.strip().split())==2:
					image,labels=line.strip().split()
					if '.' not in image:
						image+='.jpg'
					labels=int(labels)
				else:
					line=line.strip().split()
					image=line[0]
					labels=map(int,line[1:])
			img_name_list.append(os.path.join(data_dir,image))
			img_labels.append(labels)
		return img_name_list,np.array(img_labels,dtype=np.float32)

	def read_mask_list(self,mask_folder='mask'):
		"""
		read the mask list according to image list
		note that the mask image  in different folder  with the same structure of image
		/path/image_folder/train/tp/name.jpg
		/path/mask_folder/train/tp/name.png
		:return: self.mask list
		"""
		mask_lists=[]
		datepad=re.compile(r'(.*)/(.*)/(.*/.*/.*)')
		for image in self.image_list:
			m=datepad.match(image)
			mask_list=os.path.join(m.group(1),mask_folder,m.group(3))
			mask_list=mask_list.split('.')[0]+'.png'
			mask_lists.append(mask_list)
		return mask_lists




def get_name_id(name_path):
	name_id=name_path.strip().split('/')[-1]
	name_id=name_id.strip().split('.')[0]
	return name_id


if __name__=='__main__':
	import json
	import tqdm
	datalist='../datafolder/C2_MASK_ROI/rain_list.txt'

	data=DATASET(root_dir='',datalist_file=datalist)
	img_mean=np.zeros((len(data),3))
	img_std=np.zeros((len(data),3))
	for idx in tqdm.tqdm(range(len(data))):
		img,_=data[idx]
		numpy_img=np.array(img)
		per_img_mean=np.mean(numpy_img,axis=(0,1))/255.0
		per_img_std=np.std(numpy_img,axis=(0,1))/255.0
		img_mean[idx]=per_img_mean
		img_std[idx]=per_img_std
	mean=np.mean(img_mean,axis=0)
	mean=list(map(str,mean))
	std=np.mean(img_std,axis=0)
	std=list(map(str,std))
	json.dump({'mean':mean,'std':std},open('mean_std.json','w'))