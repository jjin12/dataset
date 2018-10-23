import torch
from datasets import cityscapes
from PIL import Image, ImageOps
import torchvision.transforms as myTransform
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
from erfnet import Net
from torch.optim import SGD, Adam
from PIL import Image
from torch.autograd import Variable
from tensorboardX import SummaryWriter 
from utils import evaluate_segmentation
import math
import os
import random
from transform import ToLabel, Relabel

class MyCoTransform(object):
    def __init__(self, augment=True, height=512):
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        # if(self.augment):
        #     Random hflip
        #     hflip = random.random()
        #     if (hflip < 0.5):
        #         input = input.transpose(Image.FLIP_LEFT_RIGHT)
        #         target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
        #     Random translation 0-2 pixels (fill rest with padding
        #     transX = random.randint(-2, 2) 
        #     transY = random.randint(-2, 2)

        #     input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
        #     target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
        #     input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
        #     target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)
    
def main():
    # transform = myTransform.Compose([myTransform.Resize(512, Image.BILINEAR), myTransform.ToTensor()])
    co_transform = MyCoTransform(augment=True)#1024)
    co_transform_val = MyCoTransform(augment=False)#1024)
    root = "/home/jjin/adl/cityscapes_dataset"
    dataset_train = cityscapes(root, co_transform)
    loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True,num_workers=4)
    
    dataset_val = cityscapes(root, co_transform_val, subset='val')
    loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False,num_workers=4)
    
    NUM_CLASSES = 20
    weight = torch.ones(NUM_CLASSES)
    weight[0] = 2.8149201869965
    weight[1] = 6.9850029945374
    weight[2] = 3.7890393733978
    weight[3] = 9.9428062438965
    weight[4] = 9.7702074050903
    weight[5] = 9.5110931396484
    weight[6] = 10.311357498169
    weight[7] = 10.026463508606
    weight[8] = 4.6323022842407
    weight[9] = 9.5608062744141
    weight[10] = 7.8698215484619
    weight[11] = 9.5168733596802
    weight[12] = 10.373730659485
    weight[13] = 6.6616044044495
    weight[14] = 10.260489463806
    weight[15] = 10.287888526917
    weight[16] = 10.289801597595
    weight[17] = 10.405355453491
    weight[18] = 10.138095855713
    weight[19] = 0
    weight = weight.cuda()
    
    criterion = CrossEntropyLoss2d(weight)
    print(type(criterion))
    model = Net(NUM_CLASSES).cuda()
    
    # print(model)

    optimizer = Adam(model.parameters(), 0.001, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)      ## scheduler 2
    
    savedir = '/home/jjin/adl/myImplementation/datasets/save'
    automated_log_path = savedir + "/automated_log_encoder.txt"
    modeltxtpath = savedir + "/model_encoder.txt"
    
    start_epoch = 1
    iteration = 1
       
    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                    continue
            own_state[name].copy_(param)
        return model
    weightspath = '/home/jjin/adl/myImplementation/datasets/trained_models/erfnet_pretrained.pth'
    
    for epoch in range(start_epoch, 10):
        # model.train()
        # for step, (images, labels) in enumerate(loader_train):
        #     images = images.cuda()
        #     labels = labels.cuda()
            
        #     inputs = Variable(images)
        #     targets = Variable(labels).long()
        #     outputs = model(inputs)
    
        #     optimizer.zero_grad()
        #     loss = criterion(outputs, targets[:, 0])
        #     loss.backward()
        #     optimizer.step()
        #     iteration = iteration + 1
        #     pred = outputs.data.max(1)[1].cpu().numpy().flatten()
        #     gt = labels.data.cpu().numpy().flatten()
        #     global_accuracy, class_accuracies, prec, rec, f1, iou = evaluate_segmentation(pred, gt, NUM_CLASSES)
        #     print('Epoch {} [{}/{}] Train_loss:{}'.format(epoch, step, len(loader_train), loss.data[0])) # loss.item() = loss.data[0]
        # torch.save(model.state_dict(), '{}_{}.pth'.format(os.path.join("/home/jjin/adl/myImplementation/datasets/save","model"),str(epoch)))   
        #   
        # model.load_state_dict(torch.load('/home/jjin/adl/myImplementation/datasets/trained_models/erfnet_pretrained.pth'))
        model = load_my_state_dict(model, torch.load(weightspath))
        model.eval()
        with torch.no_grad():
            for step_val, (images_val, labels_val) in enumerate(loader_val):

                images_val = images_val.cuda()
                labels_val = labels_val.cuda()

                inputs_val = Variable(images_val)   
                targets_val = Variable(labels_val).long()
                outputs_val = model(inputs_val) 

                loss_val = criterion(outputs_val, targets_val[:, 0])
              #  time_val.append(time.time() - start_time)
            
                pred = outputs_val.data.max(1)[1].cpu().numpy().flatten()
                gt = labels_val.data.cpu().numpy().flatten()
                global_accuracy, class_accuracies, prec, rec, f1, iou = evaluate_segmentation(pred, gt, NUM_CLASSES)

            print('Epoch {} [{}/{}] val_loss:{}'.format(epoch, step_val, len(loader_val), loss_val.data[0])) # loss.item() = loss.data[0]
    
if __name__ == '__main__':
    main()
