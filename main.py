import torch
from datasets import cityscapes
import torchvision.transforms as myTransform

def main():
    transform = myTransform.Compose([myTransform.ToTensor()])
    root = "/home/cityscapes_dataset"
    hi = cityscapes(root, transform)
    hi.__getitem__(10)
    
if __name__ == '__main__':
    main()