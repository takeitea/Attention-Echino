import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
folder="../datafolder/c2_mask/ROI/mask/val"

cl=folder+'/CL/*.png'
ce1=folder+'/CE1/*.png'

for image in glob.glob(cl):
	imat=cv2.imread(image).astype(np.float)
	cv2.imshow(" image",(imat*64).astype(np.int8))
	cv2.waitKey(1000)
	print(np.unique(imat))