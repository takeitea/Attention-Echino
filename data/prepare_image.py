"""
keep image spacial ratios and resize the max-side to SIZE and if the the max-side is small than SIZE,
the image will be pad with zeros
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

class ResizePadding(object):
	def __init__(self, size=224,zoom=False):
		self.size = size
		self.zoom=zoom

	def transform(self, image):
		"""
		:param image: the numpy data of image
		:return:resized image
		"""
		assert len(image.shape) == 3
		max_idx = np.argmax(image.shape[:2])
		min_idx = np.argmin(image.shape[:2])
		max_side = image.shape[max_idx]
		min_side = image.shape[min_idx]
		if max_idx > min_idx:  # [min_side,max_side]
			if max_side >= self.size:  # [max_side,max_side]
				image = np.pad(image, (((max_side - min_side) // 2, (max_side - min_side) // 2), (0, 0), (0, 0)),
							   mode='constant')
				image = cv2.resize(image, (self.size, self.size),interpolation=cv2.INTER_CUBIC)
			else:
				image = np.pad(image, (
				((self.size - min_side) // 2, (self.size - min_side) // 2), ((self.size - max_side) // 2,
																			 (self.size - max_side) // 2), (0, 0)),
							   mode='constant')
				image = cv2.resize(image, (self.size, self.size),interpolation=cv2.INTER_CUBIC)
		else:
			if max_side >= self.size: # [max_side,min_side]
				image = np.pad(image, ((0, 0), ((max_side - min_side) // 2, (max_side - min_side) // 2), (0, 0)),
							   mode='constant')
				image = cv2.resize(image, (self.size, self.size),interpolation=cv2.INTER_CUBIC)
			else:
				image = np.pad(image, (
				((self.size - max_side) // 2, (self.size - max_side) // 2), ((self.size - min_side) // 2,
																			 (self.size - min_side) // 2), (0, 0)),
							   mode='constant')
				image = cv2.resize(image, (self.size, self.size),interpolation=cv2.INTER_CUBIC)
		return image


if __name__ == '__main__':
	transform = ResizePadding()
	image1 = np.ones([255, 270, 3], dtype=np.uint8) * 255
	image2 = np.ones([85, 90, 3], dtype=np.uint8) * 255
	image3 = np.ones([305, 100, 3], dtype=np.uint8) * 255
	image4 = np.ones([105, 300, 3], dtype=np.uint8) * 255
	image5 = np.ones([95, 80, 3], dtype=np.uint8) * 255
	for img in [image1, image2, image3,image4,image5]:
		print(img.shape)
		plt.imshow(img)
		plt.show()

		print(transform.transform(img).shape)
		plt.imshow(transform.transform(img))
		plt.show()
