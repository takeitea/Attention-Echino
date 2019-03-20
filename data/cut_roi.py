"""
to cut roi base the bounding box
if the pic has multi-label cut each target and resize to the target_shape
if the pic has one target ,if the target is bigger than target_shape, keep spatial rates and pad zeros ,then resize to target_shape
if the target is small than target_shape,then zeros padding and cut into target_shape
"""
import os
import cv2
import numpy as np
from collections import defaultdict
import argparse

DATA_DIR = '/data/wen/Dataset/data_maker/classifier/c9_original/'


class Preprocess:

	def __init__(self, image_path, txt, save=DATA_DIR + 'train_roi', target_shape=256):
		self.save = save
		self.image_path = image_path
		self.txt = txt
		self.target_shape = target_shape
		self.annotation = defaultdict(list)
		self.classes = []
		self.get_anni()

	def get_anni(self):
		"""
		load annotation
		:return:
		"""
		with open(self.txt, 'r') as L:
			lines = L.readlines()
			for line in lines:
				image, tp, x1, y1, x2, y2 = line.strip().split(" ")
				box = [x1, y1, x2, y2]
				if tp not in self.classes:
					self.classes.append(tp)
				box =list (map(int, box))
				self.annotation[image].append([tp, box])

	def cut_padd(self, image, box, is_multi=False):
		padded_image = np.zeros([image.shape[0] + self.target_shape // 2, image.shape[1] + self.target_shape // 2, 3])
		padded_image[:image.shape[0], :image.shape[1], :] = image
		height = box[3] - box[1]
		width = box[2] - box[0]
		center_x = box[0] + width // 2
		center_y = box[1] + height // 2
		min_side = min(height, width)
		max_side = max(height, width)
		channel = 3
		if is_multi:
			target_roi = np.zeros([self.target_shape, self.target_shape, channel])
			target_roi[:width, :height, :channel] = image[box[0]:box[2], box[1]:box[3], :channel]
		else:

			target_roi = np.zeros([max_side, max_side, channel])
			if min_side > self.target_shape:
				target_roi[:, :, :channel] = image[box[0]:box[0] + max_side, box[1]:box[1] + max_side, :channel]
				target_roi.resize([self.target_shape, self.target_shape])

			else:
				target_roi = padded_image[center_x - self.target_shape // 2:center_x + self.target_shape // 2,
				             center_y - self.target_shape // 2: center_y + self.target_shape // 2, :channel]
				# target_roi.resize([self.target_shape, self.target_shape,channel])
		return target_roi

	def cut(self):
		# with open(self.txt, 'r')as L:
		# 	lines = L.readlines()
		# 	for line in lines:
		# 		image, tp, x1, y1, x2, y2 = line.strip().split(" ")
		# 		img = cv2.imread(os.path.join(self.image_path, image))
		# 		box = np.asarray([x1, y1, x2, y2], dtype=np.int32)
		# 		cut_image = self.cut_padd(img, box, len(self.annotation[image]) > 1)
		#
		for image in self.annotation.keys():
			img=cv2.imread(os.path.join(self.image_path,image))
			for indx,anno in enumerate(self.annotation[image]):
				tp,box=anno
				roi=self.cut_padd(img, box, len(self.annotation[image])>1)
				new_name=image.split('.')[0]+str(indx)+ '.jpg'
				cv2.imwrite(os.path.join(self.save,tp,new_name),roi)



	def mkdir(self):
		if not os.path.exists(self.save):
			os.mkdir(self.save)
			for cls in self.classes:
				os.mkdir(os.path.join(self.save,cls))
		else:
			raise RuntimeError("%s is exist" % self.save)

def main():
	args= argparse.ArgumentParser(description="cut roi")
	args.add_argument("--image_path", type=str, default=DATA_DIR + 'train')
	args.add_argument('--txt', type=str, default=DATA_DIR + 'train.txt')

	parse =args.parse_args()
	p = Preprocess(parse.image_path, parse.txt)
	# p.get_anni()
	p.mkdir()
	p.cut()



if __name__ == "__main__":
	main()
