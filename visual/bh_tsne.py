from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import h5py
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D

def imscatter(x,y,image,ax=None,zoom=1.):
	if ax is None:
		ax=plt.gca()
	try:
		image=['.'+i for i in image ]
		image=[ plt.imread(i) for i in image]
	except TypeError:
		pass
	im=[OffsetImage(i,zoom=zoom) for i in image]
	x,y=np.atleast_1d(x,y)

	artists=[]
	# for x0,y0,img in zip(x,y,im):
	# 	ab=AnnotationBbox(img,(x0,y0),xycoords='data',frameon=False)
	# 	artists.append(ax.add_artist(ab))
	# ax.update_datalim(np.column_stack([x,y]))
	ax.autoscale()
	return artists

name=['ae1','ae2','ae3','ce1','ce2','ce3','ce4','ce5','cl']
id2name={ idx:name for idx,name in enumerate(name)}

data=json.load(open("data.json",'r'))
X=data['feature']
y=data['label']


X_2d=TSNE(n_components=2,verbose=1,random_state=224,learning_rate=200).fit_transform(X)
y=np.array(y)

# with h5py.File("X_d.hdf5",'w')as f:
# 	dset=f.create_dataset("data",data=X_2d)
# X_2d=h5py.File('X_d.hdf5','r')['data']

palette = np.array(sns.color_palette('hls', 9))
f = plt.figure(figsize=(16, 16), dpi=180)
axes = f.add_subplot(111)
# axes.axis('off')

# sc=axes.scatter(X_2d[:,0],X_2d[:,1],c=palette[y])


imscatter(X_2d[:,0],X_2d[:,1],data['path'],zoom=0.1)
for i in range(9):
	xtext = np.median(X_2d[y == i, :], axis=0)
	axes.text(xtext[0], xtext[1], id2name[i],fontsize=14)
#
	axes.scatter(X_2d[y==i,0],X_2d[y==i,1],color=palette[i])
# axes.plot(X_2d[:,0],X_2d[:,1],linestytle='None')
plt.title("T-sne-train")
plt.savefig('t-sne_train.png')
plt.show()