
import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
from scipy.misc import imread, imresize, imsave
import numpy as np
import random
import torch


def is_image_file(filename):
    extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    return any(filename.endswith(extension) for extension in extensions)


class DatasetFromDir(data.Dataset):
    def __init__(self, file_path, samples, height=224, width=224):

        image_dir = file_path
        self.height = height
        self.width = width
        self.scale = 4
        self.labels = []

        image_filenames = [
            join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        for i in image_filenames:
            print(i)
            if len(self.labels) >= samples:
                break
            img = imread(i)
            try:
                H, W = img.shape[0], img.shape[1]
                label_orig = Image.fromarray(np.uint8(img))
                if H <= W:
                    if H < self.height:
                        label_orig = label_orig.resize(
                            (W * self.height // H, self.height), Image.ANTIALIAS)
                else:
                    if W < self.width:
                        label_orig = label_orig.resize(
                            (self.width, H * self.width // W), Image.ANTIALIAS)
                H, W = label_orig.size
                if H > self.height and W > self.width:
                    self.labels.append(label_orig)

                if len(self.labels) >= samples:
                    break

            except (ValueError, IndexError) as e:
                print(i)
                print(img.shape, img.dtype)
                print(e)

        print('we have {} training samples'.format(len(self.labels)))

    def __getitem__(self, index):
        while True:  # hack to make sure we have a color image we can handle...
            index = random.randint(0, len(self.labels) - 1)
            label_orig = self.labels[index]

            W, H = label_orig.size
            left = random.randint(0, W - self.width - 1)
            top = random.randint(0, H - self.height - 1)
            right = left + self.width
            bottom = top + self.height
            label = label_orig.crop((left, top, right, bottom))

            data = label.resize(
                (self.width // self.scale, self.height // self.scale), Image.ANTIALIAS)

            data = np.asarray(data)
            label = np.asarray(label)

            #  currently we work only on images with 3 channels
            if label.ndim == 3:
                if label.shape[2] != 3:
                    label = label[:, :, 0:3]
                    data = data[:, :, 0:3]
                l_width = label.shape[1]
                l_height = label.shape[0]
                d_width = data.shape[1]
                d_height = data.shape[0]

                input = torch.ByteTensor(
                    torch.ByteStorage.from_buffer(data.transpose(2, 0, 1).tobytes())).float().div(255).view(3, d_height, d_width)

                target = torch.ByteTensor(
                    torch.ByteStorage.from_buffer(label.transpose(2, 0, 1).tobytes())).float().div(255).view(3, l_height, l_width)
                return input, target

    def __len__(self):
        return len(self.labels)
