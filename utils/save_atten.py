import numpy as np
import cv2
import os
import shutil

idx2catename = {
	'voc20': ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
			  'dog', 'horse',
			  'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],

	'coco80': ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
			   'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
			   'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
			   'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
			   'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
			   'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
			   'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
			   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
			   'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
			   'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
			   'hair drier', 'toothbrush'],
	'c2': ['cl', 'ce1'],
	'c6': ['cl','ce1','ce2','ce3','ce4','ce5'],
	'c9': ['ae1', 'ae2', 'ae3', 'ce1', 'ce2', 'ce3', 'ce4', 'ce5', 'cl']}


class SAVE_ATTEN(object):
	def __init__(self, save_dir='save_bins', dataset=None):
		self.save_dir = save_dir
		if dataset is not None:
			self.idx2cate = self._get_idx2cate_dict(datasetname=dataset)
		else:
			self.idx2cate = None
		if not os.path.exists(self.save_dir):
			os.makedirs(self.save_dir)

	def save_top_5_pred_labels(self, preds, org_paths, global_step):
		img_num = np.shape(preds)[0]
		for idx in range(img_num):
			img_name = org_paths[idx].strip().split('/')[-1]
			if '.JPEG' in img_name:
				img_id = img_name[:-5]
			elif '.png' in img_name or '.jpg' in img_name:
				img_id = img_name[:-4]

			out = img_id + ' ' + ' '.join(map(str, preds[idx, :])) + '\n'
			out_file = os.path.join(self.save_dir, 'pred_labels.txt')
			if global_step == 0 and idx == 0 and os.path.exists(out_file):
				os.remove(out_file)
			with open(out_file, 'a') as f:
				f.write(out)

	def save_masked_img_batch(self, path_batch, atten_batch, label_batch):

		img_num = atten_batch.size()[0]
		for idx in range(img_num):
			atten = atten_batch[idx]
			atten = atten.cpu().data.numpy()
			label = label_batch[idx]
			label = int(label)
			self._save_masked_img(path_batch[idx], atten, label)

	def _get_idx2cate_dict(self, datasetname=None):
		if datasetname not in idx2catename.keys():
			print(" the given %s dataset category names are not available ." % (str(datasetname)))
			return None
		else:
			return {idx: cate_name for idx, cate_name in enumerate(idx2catename[datasetname])}

	def _save_masked_img(self, img_path, atten, label):
		"""
		save masked images with only one ground truth label
		:param img_path:
		:param atten:
		:param label:
		:return:
		"""
		if not os.path.isfile(img_path):
			raise 'Image not exist:%s' % (img_path)
		img = cv2.imread(img_path)
		org_size = np.shape(img)
		w = org_size[0]
		h = org_size[1]
		atten_map = atten[label, :, :]
		atten_norm = atten_map
		print(np.shape(atten_map), 'Max:', np.max(atten_map), 'Min:', np.min(atten_map))
		atten_norm = cv2.resize(atten_norm, dsize=(h, w))
		atten_norm = atten_norm * 255
		heat_map = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
		img = cv2.addWeighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)
		img_id = img_path.strip().split('/')[-1]
		img_id = img_id.strip().split('.')[0]
		save_dir = os.path.join(self.save_dir, img_id + '.png')
		cv2.imwrite(save_dir, img)

	def save_top_5_atten_maps(self, atten_fuse_batch, top_indices_batch, org_paths, topk=5):
		"""
		save top-5 localization maps for generating bboxes
		:param atten_fuse_batch: normalized last layer feature maps of size (batch_size,c,w,h)
		:param top_indices_batch:  ranked predicted labels of size (batch_size,C)
		:param org_paths:
		:param topk:
		:return:
		"""
		img_num = np.shape(atten_fuse_batch)[0]
		for idx in range(img_num):
			img_id = org_paths[idx].strip().split('/')[-1][:-4]
			for k in range(topk):
				atten_pos = top_indices_batch[idx, k]
				atten_map = atten_fuse_batch[idx, atten_pos, :, :]
				heat_map = cv2.resize(atten_map, dsize=(244, 244))
				heat_map *= 255
				save_path = os.path.join(self.save_dir, 'heat_maps', 'top%d' % (k + 1))
				if not os.path.exists(save_path):
					os.makedirs(save_path)
				save_path = os.path.join(save_path, img_id + '.png')
				cv2.imwrite(save_path, heat_map)

	def normalize_map(self, atten_map):
		min_val = np.min(atten_map)
		max_val = np.max(atten_map)
		atten_norm = (atten_map - min_val) / (max_val - min_val)
		return atten_norm

	def _add_msk2img(self, img, msk, isnorm=True):
		if np.ndim(img) == 3:
			assert np.shape(img)[:2] == np.shape(msk)
		else:
			assert np.shape(img) == np.shape(msk)
		if isnorm:
			atten_norm = self.normalize_map(msk)
		atten_norm *= 255
		heat_map = cv2.applyColorMap(atten_norm.astype(np.uint8), cv2.COLORMAP_JET)
		w_img = cv2.addWeighted(img.astype(np.uint8), 0.5, heat_map.astype(np.uint8), 0.5, 0)
		return w_img

	def _draw_text(self, pic, txt):
		font = cv2.FONT_HERSHEY_SIMPLEX
		txt = txt.strip().split('\n')
		stat_y = 30
		for t in txt:
			pic = cv2.putText(pic, t, (10, stat_y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
			stat_y += 30
		return pic

	def _mark_score_on_picture(self, pic, score_vec, label_idx):
		score = score_vec[label_idx]
		txt = '%.3f' % (score)
		pic = self._draw_text(pic, txt)
		return pic

	def get_heatmap_idxes(self, gt_label):
		label_idx = []

		if np.ndim(gt_label) == 1:
			label_idx = np.expand_dims(gt_label, axis=1).astype(np.int)
		elif np.ndim(gt_label) == 2:
			for row in gt_label:
				idxes = np.where(row[0] == 1)[0] if np.ndim(row == 2) else np.where(row == 1)[0]
				label_idx.append(idxes.tolist())
		else:
			label_idx = None
		return label_idx

	def get_map_k(self, atten, k, size=(224, 224)):
		atten_map_k = atten[k, :, :]
		atten_map_k = cv2.resize(atten_map_k, dsize=size)
		return atten_map_k

	def read_img(self, img_path, size=(224, 224)):
		img = cv2.imread(img_path)
		if img is None:
			print("Image does not exist.%s" % (img_path))
			exit(0)
		if size == (0, 0):
			size = np.shape(img)[:2]
		else:
			img = cv2.resize(img, size)
		return img, size[::-1]

	def get_masked_img(self, img_path, atten, pre_label, gt_label,
					   size=(224, 224), maps_in_dir=False, save_dir=None, only_map=False):

		assert np.ndim(atten) == 4

		save_dir = save_dir if save_dir is not None else self.save_dir

		if isinstance(img_path, list) or isinstance(img_path, tuple):
			batch_size = len(img_path)
			label_indexes = self.get_heatmap_idxes(gt_label)
			for i in range(batch_size):
				img, size = self.read_img(img_path[i], size)
				img_name = img_path[i].split('/')[-1]
				img_name = img_name.strip().split('.')[0]
				if maps_in_dir:
					img_save_dir = os.path.join(save_dir, img_name)
					os.mkdir(img_save_dir)

				for k in label_indexes[i]:
					atten_map_k = self.get_map_k(atten[i], k, size)
					msked_img = self._add_msk2img(img, atten_map_k)

					suffix = str(k)
					if only_map:
						save_img = (self.normalize_map(atten_map_k) * 255).astype(np.int)
					else:
						save_img = msked_img

					if maps_in_dir:
						cv2.imwrite(os.path.join(img_save_dir, suffix + '.png'), save_img)
					else:
						pre_ffix = pre_label[0][0]
						if self.idx2cate:
							folder = self.idx2cate[k]
							folder = folder + '_' + self.idx2cate[pre_ffix]
						else:
							folder = suffix + '_' + str(pre_ffix)

						if k == pre_ffix:
							save_dir = os.path.join(save_dir, 'match', folder)
						else:
							save_dir = os.path.join(save_dir, 'missmatch', folder)
						if not os.path.exists(save_dir):
							os.makedirs(save_dir)
						shutil.copyfile(img_path[i], os.path.join(save_dir, img_name + '_.jpg'))
						cv2.imwrite(os.path.join(save_dir, img_name + '.png'), save_img)

	def get_atten_map(self, img_path, atten, save_dir=None, size=(321, 321)):
		'''
		:param img_path:
		:param atten:
		:param size: if it is (0,0) use original image size, otherwise use the specified size.
		:param combine:
		:return:
		'''

		if save_dir is not None:
			self.save_dir = save_dir
		if isinstance(img_path, list) or isinstance(img_path, tuple):
			batch_size = len(img_path)

			for i in range(batch_size):
				atten_norm = atten[i]
				min_val = np.min(atten_norm)
				max_val = np.max(atten_norm)
				atten_norm = (atten_norm - min_val) / (max_val - min_val)
				# print np.max(atten_norm), np.min(atten_norm)
				h, w = size

				atten_norm = cv2.resize(atten_norm, dsize=(h, w))
				# atten_norm = cv2.resize(atten_norm, dsize=(w,h))
				atten_norm = atten_norm * 255

				img_name = img_path[i].split('/')[-1]
				img_name = img_name.replace('jpg', 'png')
				cv2.imwrite(os.path.join(self.save_dir, img_name), atten_norm)