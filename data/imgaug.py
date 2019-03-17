from torchvision import transforms
import albumentations
from albumentations.pytorch.transforms import ToTensor

normalize = transforms.Normalize(mean=[0.275, 0.278, 0.284],
								 std=[0.170, 0.171, 0.173])


# def albu_

def preprocess_strategy(dataset='c9'):
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
			transforms.Resize(350),
			transforms.RandomResizedCrop(331),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])
		val_transforms = transforms.Compose([
			transforms.Resize(350),
			transforms.CenterCrop(331),
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
					albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.75)]),
				albumentations.SmallestMaxSize(max_size=350),
				albumentations.Normalize(mean=[0.275, 0.278, 0.284], std=[0.170, 0.171, 0.173]),
				albumentations.RandomCrop(331, 331),
				ToTensor()])
		val_transforms=albumentations.Compose([
			albumentations.Resize(350,350),
			albumentations.CenterCrop(331,331),
			albumentations.Normalize(mean=[0.275, 0.278, 0.284], std=[0.170, 0.171, 0.173]),
			ToTensor()
		])
		target_transform = get_target_transform()
	else:
		raise KeyError("=> transform method of '{}' does not exist!".format(dataset))
	return train_transforms, val_transforms


def get_target_transform():
	pass


