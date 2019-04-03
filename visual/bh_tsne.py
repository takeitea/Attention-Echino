from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np
import h5py


name=['ae1','ae2','ae3','ce1','ce2','ce3','ce4','ce5','cl']
id2name={ idx:name for idx,name in enumerate(name)}
data=json.load(open("data.json",'r'))
X=data['feature']
y=data['label']
# X_2d=TSNE(n_components=2,verbose=1,random_state=224,learning_rate=200).fit_transform(X)
y=np.array(y)

# with h5py.File("X_d.hdf5",'w')as f:
# 	dset=f.create_dataset("data",data=X_2d)
X_2d=h5py.File('X_d.hdf5','r')['data']

palette = np.array(sns.color_palette('hls', 9))
f = plt.figure(figsize=(10, 10), dpi=180)
axes = f.add_subplot(111)
axes.axis('off')
sc=axes.scatter(X_2d[:,0],X_2d[:,1],c=palette[y])
for i in range(9):
	xtext = np.median(X_2d[y == i, :], axis=0)
	axes.text(xtext[0], xtext[1], id2name[i],fontsize=28)
	axes.scatter(X_2d[y==i,0],X_2d[y==i,1],color=palette[i])
plt.title("T-sne")
plt.savefig('t-sne.png')
plt.show()