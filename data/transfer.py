import numpy as np
import os
from prepare_image import ResizePadding
import cv2
import tqdm
from build_tfRecord import convert_to
filepath = "./c9"
newpath = "./c9_350"
transfer=ResizePadding(size=350)
name=['cl','ce1','ce2','ce3','ce4','ce5','ae1','ae2','ae3']
name2id={ name:idx for idx,name in enumerate(name)}

for root, folderes, _ in os.walk(filepath):
	for folder in folderes:
		images=[]
		labels=[]
		for _, categories, _ in tqdm.tqdm( os.walk(os.path.join(root, folder))):

			for category in categories:
				for _, _, filenames in os.walk(os.path.join(root, folder,category )):
					for filename in filenames:
						image_path = os.path.join(root,folder, category, filename)
						image=cv2.imread(image_path)
						# images.append(image)
						# labels.append(name2id[category.lower()])

						pad_image=transfer.transform(image)
						if  not os.path.exists(os.path.join(newpath,folder,category)):
							os.makedirs(os.path.join(newpath,folder,category))
						cv2.imwrite(os.path.join(newpath,folder,category,filename),pad_image)
		# images_=np.array(images)
		# labels_=np.array(labels)
		# convert_to(images_, labels_, name=folder)

