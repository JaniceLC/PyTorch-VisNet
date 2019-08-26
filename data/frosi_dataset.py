"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
from PIL import Image
from torchvision.datasets.folder import *
import numpy as np 
from torchvision import transforms, utils
import cv2
import torch
import scipy.fftpack as fp

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.h5'  
]

class FrosiDataset(DatasetFolder):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        parser.set_defaults(max_dataset_size=100000, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser 
    def __init__(self, opt):
        super(FrosiDataset, self).__init__(opt.dataroot + '_' + opt.phase, default_loader, extensions=IMG_EXTENSIONS, 
                                          transform=None,
                                          target_transform=None)
                                         # is_valid_file=None)

        self.transform = transforms.Compose([
        transforms.Resize(size = (300,400), interpolation=2),
        transforms.ToTensor()])
        self.im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0),axis=1)
    

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        path, target = self.samples[index]
        image = self.loader(path) 
        image = np.asarray(image)  # numpy image
        
        stream1 = self.im2freq(image)   
        open_cv_image = image[:, :, ::-1].copy()
        im_gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        stream2 = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
        stream2 = np.asarray(stream2) 
        
        
        remmax = lambda x: x/x.max()
        remmin = lambda x: x - np.amin(x, axis=(0,1), keepdims=True)
        touint8 = lambda x: (remmax(remmin(x))*(256-1e-4)).astype('uint8')
        stream1 = touint8(stream1)
        # out = Image.new('RGB', stream1.shape[1::-1])
        # out.putdata(map(tuple, stream1.reshape(-1, 3)))
        # stream1 = out
        image = Image.fromarray(image) 
        stream2 = Image.fromarray(stream2)
        stream1 = Image.fromarray(stream1)

        image = self.transform(image)
        stream1 = self.transform(stream1)
        stream2 = self.transform(stream2)
        # stream1 = torch.rfft(image, signal_ndim = 1, onesided=True ) 

        return {'STREAM_1': stream1, 'STREAM_2': stream2, 'STREAM_3': image , 'label': target, 'A_path': path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.samples)