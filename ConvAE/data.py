import os
import glob
import torch
import numpy as np
from numpy import newaxis
import imagecorruptions
from imagecorruptions import corrupt
import imgaug as ia
import imgaug.augmenters as iaa
from skimage import io, transform
from torch.utils.data import Dataset


class DirtyDocumentsDataset(Dataset):
    def __init__(self, dirty_dir, clean_dir, transform=None):
        self.dirty_dir = dirty_dir
        self.clean_dir = clean_dir
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(self.dirty_dir))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        train_names = sorted(glob.glob(self.dirty_dir + "*.png"))
        train_image = io.imread(train_names[idx], as_gray=True)

        clean_names = sorted(glob.glob(self.clean_dir + "*.png"))
        clean_image = io.imread(clean_names[idx], as_gray=True)

        train_image = train_image[:,:,newaxis]
        clean_image = clean_image[:,:,newaxis]

        sample = { 'train_image' : train_image, 'clean_image':clean_image}

        if self.transform:
           sample = self.transform(sample)

        return sample



class DirtyDocumentsDataset_Test(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_names = sorted(glob.glob(self.root_dir + "*.png"))
        img_name = img_names[idx]
        image = io.imread(img_name)

        image = image[:,:,newaxis]

        sample = { 'image' : image}

        if self.transform:
           sample = self.transform(sample)

        return sample


###############################################################################
# Transforms                                                                  #
###############################################################################

# Test Loader Transform classes
class Rescale_Test(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > self.output_size:
                new_h = self.output_size
            else:
                new_h = self.output_size
        else:
            new_h, new_w = self.output_size
        
        img = transform.resize(image, (new_h,new_w))

        return {'image':img}
    
class ToTensor_Test(object):
    def __call__(self, sample):
        image = sample['image']

        image = image.transpose((2,0,1))
        return { 'image' : torch.from_numpy(image)}


# Train Loader Transform classes
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
    
    def __call__(self, sample):
        train_image = sample['train_image']
        clean_image = sample['clean_image']

        h, w = train_image.shape[:2]
        if isinstance(self.output_size, int):
            if h > self.output_size:
                new_h = self.output_size
            else:
                new_h = self.output_size
        else:
            new_h, new_w = self.output_size
        
        tr_img = transform.resize(train_image, (new_h,new_w))
        cl_img = transform.resize(clean_image, (new_h,new_w))

        return {'train_image':tr_img, 'clean_image':cl_img}
    
class ToTensor(object):
    def __call__(self, sample):
        train_image = sample['train_image']
        clean_image = sample['clean_image']

        train_image = train_image.transpose((2,0,1))
        clean_image = clean_image.transpose((2,0,1))
        return { 'train_image' : torch.from_numpy(train_image),'clean_image' : torch.from_numpy(clean_image)}

class RandomCrop:
    def __init__(self, size, threshold):
        assert isinstance(size,(tuple))
        self.size = size
        self.threshold = threshold

    def __call__(self, sample):
        p = np.random.rand(1)

        if p <= self.threshold:
            clean_image = sample['clean_image']
            dirty_image = sample['train_image']

            clean_img = np.array(clean_image)
            dirty_img = np.array(dirty_image)

            x, y = clean_img.shape[:2]

            x = x - 1 - self.size[0]
            y = y - 1 - self.size[1]
            min_x = int(np.random.rand(1) * x)
            min_y = int(np.random.rand(1) * y)
            max_x = min_x + self.size[0]
            max_y = min_y + self.size[1]

            clean_crop = clean_img[min_x:max_x, min_y:max_y]
            dirty_crop = dirty_img[min_x:max_x, min_y:max_y]
        else:
            clean_crop = sample['clean_image']
            dirty_crop = sample['train_image']

        return {'clean_image' : clean_crop, 'train_image' : dirty_crop}

class ImgAugTransform:
    def __init__(self):
        self.sometimes = lambda aug: iaa.Sometimes(0.5,aug)
        self.aug = iaa.Sequential([
            self.sometimes(iaa.GaussianBlur(sigma=(0,3.0))),
            self.sometimes(iaa.AdditiveGaussianNoise()),
            self.sometimes(iaa.SaltAndPepper(0.05)),
        ])

    def __call__(self, sample):
        clean_image = sample['clean_image']
        dirty_image = sample['train_image']

        clean_img = np.array(clean_image)
        dirty_img = np.array(dirty_image)
        
        train_image = self.aug.augment_image(dirty_img)

        return {'clean_image' : clean_img, 'train_image' : train_image}


