
import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
class SegAlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B,label}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.ABC_paths = sorted(make_dataset(self.dir_ABC, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

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
        ABC_path = self.ABC_paths[index]
        ABC = Image.open(ABC_path).convert('RGB')
        # split ABC image into A, B and C
        w, h = ABC.size
        ww = int(w / 3)
        A = ABC.crop((0, 0, ww, h))
        B = ABC.crop((ww, 0, ww * 2, h))
        C = ABC.crop((ww * 2, 0, w, h))

        
        transform_params = None #get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        C_transform = transforms.ToTensor() # [0, 10, 20, ..., 70] --> [0, 10/255, 20/255, ..., 70/255]

        A = A_transform(A)
        B = B_transform(B)
        C = C_transform(C)
        # print(C.shape)
        label = C[0,:,:] 
        # print(label)
        # print(torch.min(label),torch.max(label))
        label = (label * 255)//10
        label = label.long()
        out = label # [0, 10/255, 20/255, ..., 70/255] --> [0, 1, 2, 3, 4, 5, 6, 7]
        # print(label)
        # print(torch.min(label),torch.max(label))  
        # 
        ### 测试数据格式

        # size = list(label.shape)
        # label = label.view(-1)
        # ones = torch.sparse.torch.eye(8)
        # ones = ones.index_select(0,label)
        # size.append(8)
        # out = ones.view(size)
        # out = out.long()
        # out = out.permute(2,0,1) ## N,256,256
        


        return {'A': A, 'B': B, 'label':out, 'A_paths': ABC_path, 'B_paths': ABC_path, 'C_paths': ABC_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.ABC_paths)
