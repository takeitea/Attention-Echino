import torch
import torchvision.datasets
import torchvision.transforms as trans
import torch.utils.data
from torch.utils.data import distributed

MEANS = [0.275, 0.278, 0.284]
STDS = [0.170, 0.171, 0.173]


def get_data(args):

	transform_train = trans.Compose([trans.Resize((256,256)),trans.RandomResizedCrop(224),
									 trans.RandomHorizontalFlip(),
									 trans.ToTensor(),
									 trans.Normalize(MEANS,STDS)])

	transform_val=trans.Compose([trans.Resize(256),trans.CenterCrop(224),trans.ToTensor(),trans.Normalize(MEANS,STDS)])
	trainset=torchvision.datasets.ImageFolder(root=args.dir+'train',transform=transform_train)
	valset=torchvision.datasets.ImageFolder(root=args.dir+'val',transform=transform_val)
	trainloader=torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,shuffle=True,
											num_workers=32,pin_memory=True)
	valloader=torch.utils.data.DataLoader(valset,batch_size=args.batch_size,shuffle=False,num_workers=32)
	return trainloader,valloader
