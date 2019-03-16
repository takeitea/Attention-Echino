from sklearn.metrics import classification_report,accuracy_score
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
	'c6': ['ce1','ce2','ce3','ce4','ce5','cl'],
	'c9': ['ae1', 'ae2', 'ae3', 'ce1', 'ce2', 'ce3', 'ce4', 'ce5', 'cl']}


class MY_EVALUATE:
	"""

	to evaluate the classifier performance
	"""

	def __init__(self, y_true, y_pred, data_set=None):
		self.y_true = y_true
		self.y_pred = y_pred
		if data_set is not None:
			self.labels = self._get_idx2cate_dict(data_set)
		else:
			self.labels = list(set(data_set))

	def my_confusion_matrix(self,classes,saved_dir ,normalize=False,cmap=plt.cm.Blues):

		# labels=list(set(y_true))
		conf_mat = confusion_matrix(self.y_true, self.y_pred)
		# classes=classes[unique_labels(self.y_true,self.y_pred)]
		if normalize:
			conf_mat=conf_mat.astype('float')/conf_mat.sum(axis=1)[:,np.newaxis]
			print("normalized confusion matrix")
		else:
			print("without normalization")

		print(conf_mat)
		fig,ax=plt.subplots()
		im=ax.imshow(conf_mat,interpolation='nearest',cmap=cmap)
		ax.figure.colorbar(im,ax=ax)
		ax.set(xticks=np.arange(conf_mat.shape[1]),
			   yticks=np.arange(conf_mat.shape[0]),
			   xticklabels=classes,yticklabels=classes,
			   ylabel='True label',
			   xlabel='Predicted label')

		plt.setp(ax.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor')
		fmt = '.2f' if normalize else 'd'
		thresh = conf_mat.max() / 2.
		for i in range(conf_mat.shape[0]):
			for j in range(conf_mat.shape[1]):
				ax.text(j, i, format(conf_mat[i, j], fmt),
						ha="center", va="center",
						color="white" if conf_mat[i, j] > thresh else "black")
		fig.tight_layout()
		if saved_dir:
			plt.savefig(saved_dir+'/cm.png')
		plt.show()

		return ax
	def my_acc(self):
		print(accuracy_score(self.y_true,self.y_pred))

	def my_classification_report(self):

		print(classification_report(self.y_true, self.y_pred,target_names=idx2catename['c9'],digits=3))

	def _get_idx2cate_dict(self, datasetname=None):
		if datasetname not in idx2catename.keys():
			print('The given %s dataset category names are not available. The supported are: %s' \
				  % (str(datasetname), ','.join(idx2catename.keys())))
			return None
		else:
			return {idx: cate_name for idx, cate_name in enumerate(idx2catename[datasetname])}


def main():
	Root_dir = './ResNet18base_line'
	pred_txt = os.path.join(Root_dir, "result.txt")
	test_txt = os.path.join(Root_dir, "test_list.txt")
	dataset = "c6"

	pred = [line.strip().split(" ")[1] for line in open(pred_txt, 'r').readlines()]
	true = [line.strip().split(",")[-1] for line in open(test_txt, 'r').readlines()]

	my_eval = MY_EVALUATE(y_true=true, y_pred=pred, data_set=dataset)
	my_eval.my_classification_report()
	my_eval.my_confusion_matrix(idx2catename['c9'],Root_dir,normalize=True)
	my_eval.my_acc()

if __name__ == "__main__":
	main()
