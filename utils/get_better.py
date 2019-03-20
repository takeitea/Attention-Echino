import shutil
import glob
import os


v0='snapshot_c9/*/*/*.*'
v1='snapshot_c9_v1/*/*/*.*'

new='./new'
if  os.path.exists(new):
	shutil.rmtree(new)

os.mkdir(new)

files_v0=[ file for file in glob.glob(v0)]
files_v1=[ file for file in glob.glob(v1)]

for file_v1 in files_v1:
	image_name=file_v1.split('/')[-1]
	if 'missmatch' not in file_v1:
		for file_v0 in files_v0:
			if image_name in file_v0 and 'missmatch' in file_v0 :
				for file in [file_v1,file_v0]:
					sr_folder=file
					folder=file[len(file.split('/')[0]):]
					folder=folder[1:-len(folder.split('/')[-1])]
					folder=os.path.join(new,folder)
					if not os.path.exists(folder):
						os.makedirs(folder)

					shutil.copy(sr_folder,folder)
