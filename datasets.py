import torch
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
   
class cityscapes(Dataset):      
    def __init__(self, root, transform=None, subset='train'):
        self.images_root = os.path.join(root, "leftImg8bit/")
        self.labels_root = os.path.join(root, "gtFine/")
        
        self.images_root += subset
        self.labels_root += subset
            
        self.image_filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn]
        self.image_filenames.sort()
       
        self.label_filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if         f.endswith("_labelTrainIds.png")]
        self.label_filenames.sort()
        
        self.transform = transform
        
    def __getitem__(self, idx):
        image = Image.open(self.image_filenames[idx])
        image.show()
        label = Image.open(self.label_filenames[idx])
        if self.transform is not None:
            image, label = self.transform(image, label)  
        return image, label
    def __len__(self):
        return len(self.image_filenames)
        