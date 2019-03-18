from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.multiclass import unique_labels
import os
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import confusion_matrix

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
	'c6': ['ce1', 'ce2', 'ce3', 'ce4', 'ce5', 'cl'],
	'c9': ['ae1', 'ae2', 'ae3', 'ce1', 'ce2', 'ce3', 'ce4', 'ce5', 'cl']}


class MY_EVALUATE:
	"""

	to evaluate the classifier performance
	"""

	def __init__(self, Root_dir, data_set=None):
		self.Root_dir = Root_dir
		pred_txt = os.path.join(Root_dir, "result.txt")
		test_txt = os.path.join(Root_dir, "test_list.txt")
		self.y_pred= [line.strip().split(" ")[1] for line in open(pred_txt, 'r').readlines()]
		self.y_true= [line.strip().split(",")[-1] for line in open(test_txt, 'r').readlines()]

		self.pred_label={line.strip().split(' ')[0].split('/')[-1]:line.strip().split(" ")[1] for line in open(pred_txt,'r').readlines()}
		self.true_label={line.strip().split(',')[0].split('/')[-1]:line.strip().split(",")[1] for line in open(test_txt,'r').readlines()}
		self.data_set = data_set if data_set else 'c9'
		self.labels = idx2catename[self.data_set]
		self.id2cate=self._get_idx2cate_dict(self.data_set)

	def my_confusion_matrix(self, normalize=False, cmap=plt.cm.Blues):

		conf_mat = confusion_matrix(self.y_true, self.y_pred)
		classes = self.labels
		if normalize:
			conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
			print("normalized confusion matrix")
		else:
			print("without normalization")

		print(conf_mat)
		fig, ax = plt.subplots()
		im = ax.imshow(conf_mat, interpolation='nearest', cmap=cmap)
		ax.figure.colorbar(im, ax=ax)
		ax.set(xticks=np.arange(conf_mat.shape[1]),
			   yticks=np.arange(conf_mat.shape[0]),
			   xticklabels=classes, yticklabels=classes,
			   ylabel='True label',
			   xlabel='Predicted label')

		plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
		fmt = '.2f' if normalize else 'd'
		thresh = conf_mat.max() / 2.
		for i in range(conf_mat.shape[0]):
			for j in range(conf_mat.shape[1]):
				ax.text(j, i, format(conf_mat[i, j], fmt),
						ha="center", va="center",
						color="white" if conf_mat[i, j] > thresh else "black")
		fig.tight_layout()
		if self.Root_dir:
			plt.savefig(self.Root_dir + '/cm.png')
		plt.show()
		return ax

	def my_acc(self):
		print(accuracy_score(self.y_true, self.y_pred))

	def my_classification_report(self):

		print(classification_report(self.y_true, self.y_pred, target_names=idx2catename[self.data_set], digits=3))

	def _get_idx2cate_dict(self, datasetname=None):
		if datasetname not in idx2catename.keys():
			print('The given %s dataset category names are not available. The supported are: %s' \
				  % (str(datasetname), ','.join(idx2catename.keys())))
			return None
		else:
			return {idx: cate_name for idx, cate_name in enumerate(idx2catename[datasetname])}

	def copy_errors(self):
		pass


def main():
	Root_dir = './ResNet18_resize'
	my_eval = MY_EVALUATE(Root_dir)
	my_eval.my_classification_report()
	my_eval.my_confusion_matrix(normalize=True)
	my_eval.my_acc()


if __name__ == "__main__":
	main()
