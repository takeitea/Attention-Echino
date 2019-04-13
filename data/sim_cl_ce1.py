import cv2
import numpy as np
import shutil
import os
import glob

saved_path='fake_data'
sr_path='../datafolder/c2_mask/ROI/fake_image'

def get_cl(n):
	for i in range(n):
		srimage=np.ones([224,224,3]).astype(np.uint8())
		cv2.circle(srimage,(np.random.randint(50,200),np.random.randint(50,200)),np.random.randint(20,100),
				   color=(np.random.randint(255),np.random.randint(255),np.random.randint(255)),
				   thickness=np.random.randint(1,7))
		# cv2.imshow("sr",srimage)
		cv2.imwrite(saved_path+'/cl/'+str(i)+'.jpg',srimage)
		# cv2.waitKey(2000)


def get_ce1(n):
	for i in range(n):
		srimage=np.ones([224,224,3]).astype(np.uint8())
		cv2.circle(srimage,(np.random.randint(50,200),np.random.randint(50,200)),np.random.randint(20,100),
				   color=(np.random.randint(255),np.random.randint(255),np.random.randint(255)),
				   thickness=np.random.randint(5,10))
		# cv2.imshow("sr",srimage)
		cv2.imwrite(saved_path+'/ce1/'+str(i)+'.jpg',srimage)
		# cv2.waitKey(2000)

def draw_circle(image):

	srimage=cv2.imread(image)
	if 'CL' in image:
		cv2.circle(srimage, (np.random.randint(1, srimage.shape[0]), np.random.randint(1, srimage.shape[1])), np.random.randint(20, 50),
			   color=(np.random.randint(50,255), np.random.randint(50,255), np.random.randint(50,255)),
			   thickness=np.random.randint(1, 5))
	elif 'CE1'  in image:
		cv2.circle(srimage, (np.random.randint(1, srimage.shape[0]), np.random.randint(1, srimage.shape[1])), np.random.randint(20, 50),
				   color=(np.random.randint(50,255), np.random.randint(50,255), np.random.randint(50,255)),
				   thickness=np.random.randint(6, 10))
	# os.remove(image)
	cv2.imwrite(image,srimage)




# cv2.imshow("sr",srimage)



def main():
	# if os.path.exists(saved_path):
	# 	shutil.rmtree(saved_path)
	# os.makedirs(saved_path)
	# os.makedirs(saved_path + '/cl')
	# os.makedirs(saved_path + '/ce1')
	# get_cl(1000)
	# get_ce1(1000)
	for image in glob.glob(sr_path+'/*/*/*.jpg'):
		draw_circle(image)


if __name__ == '__main__':
	main()