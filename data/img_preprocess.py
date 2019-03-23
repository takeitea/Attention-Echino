from torchvision import transforms
import albumentations
from albumentations.pytorch.transforms import ToTensor
import imgaug as ia
import numpy as np
import torch
from imgaug import augmenters as iaa
import cv2
normalize = transforms.Normalize(mean=[0.275, 0.278, 0.284],
								 std=[0.170, 0.171, 0.173])


def preprocess_strategy(dataset='ImageNet'):
	if dataset.startswith('CUB'):
		train_transforms = transforms.Compose([
			transforms.Resize(448),
			transforms.CenterCrop(448),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])
		val_transforms = transforms.Compose([
			transforms.Resize(448),
			transforms.CenterCrop(448),
			transforms.ToTensor(),
			normalize,
		])
	elif dataset.startswith('Aircraft'):
		train_transforms = transforms.Compose([
			transforms.Resize((512, 512)),
			transforms.CenterCrop(448),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])
		val_transforms = transforms.Compose([
			transforms.Resize((512, 512)),
			transforms.CenterCrop(448),
			transforms.ToTensor(),
			normalize,
		])
	elif dataset.startswith('Cars'):
		train_transforms = transforms.Compose([
			transforms.Resize((448, 448)),
			transforms.CenterCrop(448),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])
		val_transforms = transforms.Compose([
			transforms.Resize(448),
			transforms.CenterCrop(448),
			transforms.ToTensor(),
			normalize,
		])
	elif dataset.startswith('ImageNet'):
		train_transforms = transforms.Compose([
			transforms.Resize(250),
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])
		val_transforms = transforms.Compose([
			transforms.Resize(250),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			normalize,
		])
	elif dataset.startswith('c9'):
		train_transforms = \
			albumentations.Compose([
				albumentations.OneOf([
					albumentations.CLAHE(clip_limit=2),
					albumentations.RandomGamma(),
					albumentations.NoOp(),
					albumentations.RandomBrightnessContrast()]),
				albumentations.OneOf([
					albumentations.ElasticTransform(),
					albumentations.MotionBlur(),
					albumentations.Cutout(),
					albumentations.NoOp()]),
				albumentations.OneOf([
					albumentations.RandomScale(),
					albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=10, p=0.75)]),
				albumentations.SmallestMaxSize(max_size=250),
				albumentations.Normalize(mean=[0.275, 0.278, 0.284], std=[0.170, 0.171, 0.173]),
				albumentations.RandomCrop(224, 224),
				ToTensor()])
		val_transforms = albumentations.Compose([
			albumentations.Resize(250, 250),
			albumentations.CenterCrop(224, 224),
			albumentations.Normalize(mean=[0.275, 0.278, 0.284], std=[0.170, 0.171, 0.173]),
			ToTensor()
		])
		target_transform = get_target_transform()
	else:
		raise KeyError("=> transform method of '{}' does not exist!".format(dataset))
	return train_transforms, val_transforms


def preprocess_strategy_with_mask():
	return ImgAugTransform(input_size=250, crop_size=224, )


def get_target_transform():
	pass


class ImgAugTransform:
	def __init__(self, input_size, crop_size):
		self.aug = iaa.Sequential([
			iaa.Scale({"height": input_size, "width": input_size}),
			iaa.Crop(px=input_size - crop_size)

		])
		self.normalize = transforms.Compose([transforms.ToTensor(), normalize])
		self.ToTensor = transforms.Compose([transforms.ToTensor()])

	def __call__(self, img, mask):
		seg_det = self.aug.to_deterministic()
		img = np.asarray(img)
		mask = np.asarray(mask)
		mask = np.where(mask > 0, 1, mask)

		# seg_on_img = ia.SegmentationMapOnImage(mask, shape=img.shape, nb_classes=2)
		#
		aug_img = seg_det.augment_image(img)
		aug_mask=seg_det.augment_image(mask)
		aug_norm = self.normalize(aug_img)
		# aug_mask = seg_det.augment_segmentation_maps([seg_on_img])[0].arr
		aug_mask =cv2.resize(aug_mask,(128,128))
		# cv2.imshow("aug_mask",aug_mask*255)
		# cv2.waitKey(2000)

		aug_mask = torch.from_numpy(aug_mask.astype(np.float32))
		return aug_norm, aug_mask
