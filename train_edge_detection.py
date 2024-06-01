import os
import sys
import time
import shutil
import random
import numpy as np
import torchnet as tnt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.autograd import Function
from torch.utils import data
import torch.distributed as dist
from datetime import datetime
from tqdm import tqdm
import socket

import model_RW as model_RW
import basic_function as func
import dataset
import transform_contour as transform_contour
import loss as loss

from IPython.core import debugger
debug = debugger.Pdb().set_trace

from tensorboardX import SummaryWriter
writer=SummaryWriter()

parserWarpper = func.MyArgumentParser()
parser = parserWarpper.get_parser()
args = parser.parse_args()

opt_manualSeed = 1000
print("Random Seed: ", opt_manualSeed)
np.random.seed(opt_manualSeed)
random.seed(opt_manualSeed)
torch.manual_seed(opt_manualSeed)
torch.cuda.manual_seed_all(opt_manualSeed)

#cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = False
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


class Trainer():
    def __init__(self, args):
        self.args = args
        self.date = datetime.now().strftime('%b%d_%H-%M-%S')+'_'+socket.gethostname()#+args.train_path
        self.best_pred = 0
        
        
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        train_transform = transform_contour.Compose([
            transform_contour.RandScale([0.5, 2.0]),
            transform_contour.RandRotate([-10, 10], padding=mean, ignore_label=255),
            transform_contour.RandomGaussianBlur(),
            transform_contour.RandomHorizontalFlip(),
            transform_contour.Crop([709, 709], crop_type='rand', padding=mean, ignore_label=255),
            transform_contour.ToTensor(),
            transform_contour.Normalize(mean=mean, std=std)])
        train_dataset = dataset.Sem_ContourData(split='train', data = args.dataset, data_root=args.dataset_path, data_list='train.txt', transform=train_transform, path_lab = 'superpixels', path_contour= args.train_path)

        self.train_loader = data.DataLoader(train_dataset, num_workers=args.workers, batch_size=args.batchsize, shuffle=True, pin_memory=True)

        resnet = models.__dict__['resnet' + str(args.layers)](pretrained=True)
        self.model = model_RW.Res_Deeplab(num_classes=args.numclasses, layers=args.layers, shrink_factor=args.shrink_factor)

        self.model.model_sed = func.param_restore(self.model.model_sed, resnet.state_dict())

        max_step = args.epochs * len(self.train_loader)
        
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 10 * ((1.0-float(step)/max_step)**0.9))
        self.criterion_CE = loss.bce2d()
        self.criterion_Entropy = loss.EdgeEntropyLoss(args.numclasses, args.sysrand, delta_weight = args.delta_weight , rho_weight =  args.rho_weight)
        self.criterion = loss.SegLoss(255)
        self.criterion_MSE = loss.EdgeMSELoss()
        '''
        self.criterion_edge = loss.EdgeLoss()
        self.criterion_entropy = loss.EntropyLoss()
        self.criterion_multiclass = loss.MultiClassLoss(reduction='max')
        self.criterion_intraclass = loss.EdgeLoss_discriminate(edge_balance=True)
        '''
        
        self.model = torch.nn.DataParallel(self.model)
        self.model = self.model.cuda()

    def train(self, epoch):
        self.model.train()
        losses = func.AverageMeter()
        tbar = tqdm(self.train_loader)
        for i, batch in enumerate(tbar):
            cur_lr = self.scheduler.get_lr()[0]
            img, label, edge, path_name = batch

            #edge[(label==255) & (edge == 0)]=255
            gt = torch.unsqueeze(edge,dim=1).float()
            
            batch_size = img.size()[0]
            input_size = img.size()[2:4]
            
            img_v = img.cuda(non_blocking=True)
            gt_v =gt.cuda(non_blocking=True)
            label_v = label.cuda(non_blocking=True)
            edge_v =edge.cuda(non_blocking=True)
            #adj_v = adj.cuda(non_blocking=True)

            pred = self.model(img_v)

            pred_sg_up = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
            
            #loss = self.criterion(pred_sg_up, label_v.squeeze(1))

            if args.numclasses!=1:
                loss=self.criterion_Entropy(pred_sg_up, gt_v)
                #loss=self.criterion(pred_sg_up,label_v)
            elif args.losstype=='BCE':
                loss=self.criterion_CE(torch.sigmoid(pred_sg_up), gt_v)
            elif args.losstype=='MSE':
                loss=self.criterion_MSE(torch.sigmoid(pred_sg_up), gt_v)

            losses.update(loss.item(), img.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            tbar.set_description('Train [{0}] Loss {loss.val:.3f} {loss.avg:.3f} {lr:.5f} Best {best:.4f}'.format(epoch, loss=losses, lr=cur_lr, best=self.best_pred))
        writer.add_scalar('train/train_loss',losses.avg,epoch)


    def validate_tnt(self, epoch):
        class_pixel = torch.zeros(args.numclasses)
        total_pixel = torch.zeros(args.numclasses)
        losses = func.AverageMeter()
        tbar = tqdm(self.val_loader)
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tbar):
                img, gt, edge, _ = batch

                batch_size = img.size()[0]
                input_size = img.size()[2:4]
                
                img_v = img.cuda(non_blocking=True)
                gt_v = gt.cuda(non_blocking=True)
                edge_v = edge.cuda ( non_blocking=True )

                pred = self.model(img_v)

                pred_sg_up = F.interpolate(pred, size=input_size, mode='bilinear', align_corners=True)
                pred_sg_up_label = torch.max(pred_sg_up, 1)[1]

                loss_edge = -self.criterion_entropy ( pred_sg_up, edge_v, 1 )
                loss_nonedge = self.criterion_entropy ( pred_sg_up, edge_v, 0 )
                loss = loss_edge + loss_nonedge
                
                for i in range(args.numclasses):
                    for j in range(batch_size):
                        gt_=gt[j]
                        pred_sg_up_label_=pred_sg_up_label[j]
                        gt_mask=(gt_==i)
                        if torch.sum(gt_mask)!=0:
                            _,count=torch.unique(pred_sg_up_label_[gt_mask],return_counts=True)
                            class_pixel[i] = class_pixel[i] + torch.max(count)
                            total_pixel[i] = total_pixel[i] + torch.sum(gt_mask)


                losses.update(loss.item(), img.size(0))
                tbar.set_description('Valid [{0}] Loss {loss.val:.3f} {loss.avg:.3f} E {eloss:.3f} N {nloss:.3f}'.format(
                    epoch, loss=losses, eloss=loss_edge, nloss=loss_nonedge))

            mean_acc_ind = class_pixel/total_pixel
            mean_acc_all = torch.mean(mean_acc_ind)
            print(' * ACC_All {acc}'.format(acc=mean_acc_all))
            print(' * ACC_Ind {acc}'.format(acc=mean_acc_ind))
            writer.add_scalar('val/val_loss',losses.avg,epoch)
            #writer.add_scalar('val/val_iou',mean_iou_all,epoch)

        return mean_acc_all, mean_acc_ind

trainer = Trainer(args)
#trainer.validate_tnt(0)
for epoch in range(args.epochs):

    # train and validate
    trainer.train(epoch)

    func.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': trainer.model.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
    }, trainer.date, True, trainer.args.shfilename)

        
writer.close()

for item in args.__dict__.items():
    print(item)