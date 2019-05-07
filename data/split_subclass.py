import os


subclass0=['CL','CE1']
subclass1=['CE2','CE3','CE4']
subclass2=['AE1','AE2','AE3']

root='/data/wen/data/C9/'

train='train.txt'
test='test.txt'
sub_train=['test_sub'+str(i)+'.txt' for i in range(3)]


for line in open(root+test,'r').readlines():
	tp=line.split(' ')[1]
	if tp in subclass0:
		T=open(root+sub_train[0],'a')
	elif tp in subclass1:
		T=open(root+sub_train[1],'a')
	elif tp in subclass2:
		T=open(root+sub_train[2],'a')
	else:
		T=open(root+'CE5_test.txt','a')
	T.writelines(line)

