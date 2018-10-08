import torch
from datasets import cityscapes
import torchvision.transforms as myTransform
from torch.utils.data import Dataset, DataLoader
from erfnet import Net
from torch.optim import SGD, Adam
from PIL import Image
from torch.autograd import Variable
from tensorboardX import SummaryWriter 
from metrics import runningScore, averageMeter

class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)

def main():
    transform = myTransform.Compose([myTransform.Resize(512, Image.BILINEAR), myTransform.ToTensor()])
    root = "/home/jjin/adl/cityscapes_dataset"
    dataset_train = cityscapes(root, transform)
    loader_train = DataLoader(dataset_train, batch_size=4, shuffle=True,num_workers=4)
    
    dataset_val = cityscapes(root, transform, subset='val')
    loader_val = DataLoader(dataset_val, batch_size=4, shuffle=False,num_workers=4)
    
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
    
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)      ## scheduler 2
    
    savedir = '/home/jjin/adl/myImplementation/datasets/save'
    automated_log_path = savedir + "/automated_log_encoder.txt"
    modeltxtpath = savedir + "/model_encoder.txt"
    
    writer = SummaryWriter('/home/jjin/adl/myImplementation/datasets/runs')
    start_epoch = 1
    iteration = 1
    
    # Setup Metrics
    running_metrics_val = runningScore(NUM_CLASSES)
    val_loss_meter = averageMeter()
    time_meter = averageMeter()
    
    for epoch in range(start_epoch, 3):
        model.train()
        for step, (images, labels) in enumerate(loader_train):
            images = images.cuda()
            labels = labels.cuda()
            
            inputs = Variable(images)
            targets = Variable(labels).long()
            outputs = model(inputs)
    
            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()
            iteration = iteration + 1
            print('Epoch {} [{}/{}] Train_loss:{}'.format(epoch, step, len(loader_train), loss.data[0])) # loss.item() = loss.data[0]
            writer.add_scalar('loss/train_loss', loss.data[0], iteration)
             
        
        model.eval()
        with torch.no_grad():
            for step_val, (images_val, labels_val) in enumerate(loader_val):

                images_val = images_val.cuda()
                labels_val = labels_val.cuda()

                inputs_val = Variable(images_val)    #volatile flag makes it free backward or outputs for eval
                targets_val = Variable(labels_val).long()
                outputs_val = model(inputs_val) 

                loss_val = criterion(outputs_val, targets_val[:, 0])
              #  time_val.append(time.time() - start_time)


                pred = outputs_val.data.max(1)[1].cpu().numpy()
                gt = labels_val.data.cpu().numpy()


                running_metrics_val.update(gt, pred)
                val_loss_meter.update(loss_val.item())

            writer.add_scalar('loss/val_loss', val_loss_meter.avg,  iteration)
            score, class_iou = running_metrics_val.get_scores()
            print('Epoch {} [{}/{}] val_loss:{}'.format(epoch, step, len(loader_val), loss_val.data[0])) # loss.item() = loss.data[0]
            print("score", score, "class_iou", class_iou)
    writer.close()
    
if __name__ == '__main__':
    main()