import os
import numpy
import shutil
import glob
import  numpy as np
imagefolder='../datafolder/inature/*/*/*.jpg'
train='../datafolder/inature/train/'
val='../datafolder/inature/val'
all='../datafolder/inature/all'

all_classes=[]
os.mkdir(train)
os.mkdir(val)
os.mkdir(all)
for image_path in glob.glob(imagefolder):
	classes,subfolder,image=image_path.split('/')[-3:]
	all_classes.append(classes)
all_classes=set(all_classes)

for folder in [train,val,all]:
	for classes in all_classes:
		os.mkdir(os.path.join(folder,classes ))

for image_path in glob.glob(imagefolder):
	classes,subfolder,image=image_path.split('/')[-3:]
	print(classes)
	shutil.copy(image_path,os.path.join(all,classes,image))
	if np.random.randn()>0.8:
		shutil.copy(image_path,os.path.join(val,classes,image))
	else:
		shutil.copy(image_path,os.path.join(train,classes,image))

for classes in all_classes:
	shutil.rmtree('../datafolder/inature/'+classes)