import numpy as np
import cv2
import os
from PIL import Image
from numpy.random import shuffle
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)
def cv_loader( path,gray=False):
    """

    :param path:
    :return: RGB channel images
    """
    if gray:
        imgmat = cv2.imread(path,0)
    else:
        imgmat = cv2.imread(path)[:, :, ::-1]
    return np.array(imgmat)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class C9:

    def __init__(self, root='../datafolder/C2_MASK_ROI/ROI/image/val', dims=[215, 215, 1],
                 saved_path='./h5', extensions=IMG_EXTENSIONS, transform=None,
                 loader=cv_loader,gray=True,
                 target_transform=None):
        self.dims = dims
        self.loader = loader
        self.saved_path = saved_path
        self.gray=gray
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(
                extensions)))

        self.root = root
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path,self.gray)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)



    def plot_size_distibult(self):
        pass

    def load_c9(self, count=None,is_shuffle=True):

        data_count = count if count is not None and count < self.__len__() else self.__len__()
        data_index=[i for i in range(self.__len__())]
        if is_shuffle:
            shuffle(data_index)
        data_index=data_index[:data_count]
        data = {"data": [], "target": []}

        for i in data_index:
            img, label = self.__getitem__(i)
            data["data"].append(img)
            data["target"].append(label)
        data["data"]=np.asarray(data["data"])
        data["target"]=np.asarray(data["target"])
        return data
    def write_txt(self,tp='train'):
        with open(tp+'_list.txt','w')as L:
            for item in self.samples:
                print(item[0])
                L.writelines([item[0],",",str(item[1])+'\n'])

def main():
    c9 = C9(loader=cv_loader)
    data = c9.load_c9(is_shuffle=False)

    print(data["data"].shape)
    c9.write_txt('val')

if __name__ == '__main__':
    main()
