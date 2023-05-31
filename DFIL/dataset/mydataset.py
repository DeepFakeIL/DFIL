"""
Author: Honggu Liu

"""

from PIL import Image
from torch.utils.data import Dataset
import os
import random
import glob

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None,aug = False, get_feature = False, add_memory = False):

        imgs = []

        if get_feature :
            
            fh = open(txt_path, 'r')
            for line in fh:
                line = line.rstrip()
                words = line.split(',')
                #print(words)
                imgs.append((words[1], int(words[0])))
        elif add_memory:
            print("????")
            fh = open(txt_path, 'r')
            for line in fh:
                line = line.rstrip()
                words = line.split(',')
                image_file = words[1]
                path = os.path.join(image_file,'*.png')
                image_filenames = sorted(glob.glob(path))
                for i in image_filenames:
                    imgs.append((i, int(words[0])))
            
            
            txt_path = 'Deepfake-Detection-master/Deepfakes_Memory_set/sum_20221215.txt'
            fh = open(txt_path, 'r')
            for line in fh:
                line = line.rstrip()
                words = line.split(',')
                imgs.append((words[1], int(words[0])))
    
        else:
            print("????")
            fh = open(txt_path, 'r')
            for line in fh:
                line = line.rstrip()
                words = line.split(',')
                #print(words[0],words[1])
                # print(words)
                # print (words[2].rjust(5,'0'))

                image_file = words[1]
                path = os.path.join(image_file,'*.png')
                image_filenames = sorted(glob.glob(path))
                #print(image_filenames)

                #imgs.append((words[1]+'/out'+words[2].rjust(5,'0')+'.png', int(words[0])))
                for i in image_filenames:
                    imgs.append((i, int(words[0])))

        print("the number of images: ",len(imgs))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

