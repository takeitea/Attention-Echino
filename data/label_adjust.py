import numpy as np
import glob
import cv2
import  tqdm

label_path="/data/wen/data/c9_mask/train/label/*.png"
for impath in tqdm.tqdm( glob.glob(label_path)):
	img=cv2.imread(impath)
	img[img==128]=1
	img[img==255]=2
	cv2.imwrite(impath,img)

	# print(' {}'.format(set(img).sort()