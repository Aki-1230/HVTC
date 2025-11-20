
import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
class SegDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B,label}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        
        SegDataset
          |---train_img
          |---train_label
          |---test_img
          |---test_label

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_img = os.path.join(opt.dataroot, opt.phase + '_img')  # get the image directory
        self.dir_label = os.path.join(opt.dataroot, opt.phase + '_label')  # get the label directory
        self.img_paths = sorted(make_dataset(self.dir_img, opt.max_dataset_size))  # get image paths
        self.label_paths = sorted(make_dataset(self.dir_label, opt.max_dataset_size))  # get label paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.transform_img = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))
        self.transform_label = get_transform(self.opt, grayscale=(self.opt.output_nc == 1))


    def get_one_hot(label,N):
        ## label [3,256,256]
    
        size = list(label.size)
        label = label.view(-1)
        ones = torch.sparse.torch.eye(N)
        ones = ones.index_select(0,label)
        size.append(N)
        out = ones.view(size)
        out = out.permute(2,0,1)
        return out

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            C (tensor) - - label for image
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
            C_paths (str) - - image paths 
        """
        # read a image given a random integer index
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert('RGB')
        label_path = self.label_paths[index]
        label = Image.open(label_path).convert('L')

        img_transform = get_transform(self.opt, grayscale=(self.opt.input_nc == 1))
        label_transform = transforms.ToTensor() # [0, 10, 20, ..., 70] --> [0, 10/255, 20/255, ..., 70/255]

        img = img_transform(img)
        label = label_transform(label)
        label = (label * 255)//10
        label = label.long() # [0, 10/255, 20/255, ..., 70/255] --> [0, 1, 2, 3, 4, 5, 6, 7]

        return {'img': img, 'label':label, 'img_paths': img_path, 'label_paths': label_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)
