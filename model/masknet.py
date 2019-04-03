from .resnet_mask import resnet18_mask
from .resnet import resnet18
from torch.nn.modules import Module
class Masknet18(Module):
	def __init__(self):
		super(Masknet18,self).__init__()
		self.masknet=resnet18_mask(pretrained=True)
		self.classnet=resnet18(pretrained=True,num_classes=2)

	def forward(self,x):
		full_image=x
		mask=self.masknet(full_image)

		output1,_,_=self.classnet(full_image)
		output2,_,_=self.classnet(full_image*mask)
		return output1,output2
