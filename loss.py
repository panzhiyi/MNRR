import os
import sys
import time
import shutil
import random
import argparse
import numpy as np
import torchnet as tnt
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, Function
from torch.utils import data

from IPython.core import debugger
debug = debugger.Pdb().set_trace

# label = 255 is ambiguious label, and only some gts have this label.
class SegLoss ( nn.Module ) :
    def __init__(self, ignore_label=255, mode=1) :
        super ( SegLoss, self ).__init__ ()
        if mode == 1 :
            self.obj = torch.nn.CrossEntropyLoss ( ignore_index=ignore_label )
        else :
            self.obj = torch.nn.NLLLoss2d ( ignore_index=ignore_label )

    def __call__(self, pred, label) :
        loss = self.obj ( pred, label )
        return loss

class bce2d ( nn.Module ) :
    def __init__(self) :
        super ( bce2d, self ).__init__ ()

    def __call__(self, input, target) :
        n, c, h, w = input.size()
        # assert(max(target) == 1)
        mask_edge = (target==1)
        mask_nonedge = (target==0)
        nonedge_weight =  torch.sum(mask_edge.float()) / (torch.sum(mask_edge.float())+torch.sum(mask_nonedge.float()))
        edge_weight = 1 - nonedge_weight
        weight =  torch.zeros(target.size()).cuda()
        weight[target==1]=edge_weight
        weight[target==0]=nonedge_weight
        loss = F.binary_cross_entropy(input, target, weight, reduction='mean')
        return loss

class EdgeMSELoss ( nn.Module ) :
    def __init__(self) :
        super ( EdgeMSELoss, self ).__init__ ()
        self.mseloss=nn.MSELoss(reduction='none')

    def __call__(self, input, target) :
        n, c, h, w = input.size()
        mask = (target!=255)
        loss = torch.sum(self.mseloss(input, target).mul(mask)) / torch.sum ( mask )
        return loss

class EdgeEntropyLoss ( nn.Module ) :
    def __init__(self, class_num = 20, type = 'Entropy', delta_weight = 0.25, rho_weight = 0.25) :
        super ( EdgeEntropyLoss, self ).__init__ ()
        self.type = type
        self.class_num = class_num
        self.delta_weight=delta_weight
        self.rho_weight=rho_weight
        if self.type=='Entropy':
            self.delta = (1 - delta_weight) * math.log(class_num)
            self.rho = rho_weight * math.log(class_num)
        elif self.type == 'Std':
            self.delta = (1 - delta_weight) * math.sqrt(class_num-1)/ class_num
            self.rho = rho_weight * math.sqrt(class_num-1)/ class_num
        elif self.type == 'Gini':
            self.delta = (1 - delta_weight) * (class_num-1)/class_num
            self.rho = rho_weight * (class_num-1)/class_num

    def __call__(self, input, target) :
        n, c, h, w = input.size()
        target = torch.squeeze(target)
        # assert(max(target) == 1)
        mask_edge = (target==1)
        mask_nonedge = (target==0)
        if self.type == 'Entropy':
            b = F.softmax ( input, dim=1 ) * F.log_softmax ( input, dim=1 )
            b = -1.0 * torch.sum ( b, dim=1 )
        elif self.type == 'Std':
            b = F.softmax ( input, dim=1 )
            b = self.rho / self.rho_weight - torch.std( b, dim=1 )
        elif self.type == 'Gini':
            b = F.softmax ( input, dim=1 )
            b = 1 - torch.sum(b.pow(2) ,dim=1)
        entropy_edge = b.mul ( mask_edge )
        entropy_nonedge = b.mul( mask_nonedge )
        entropy_edge = torch.clamp ( entropy_edge , max= self.delta )
        entropy_nonedge = torch.clamp ( entropy_nonedge , min= self.rho )
        loss = torch.sum ( entropy_nonedge ) / (torch.sum ( mask_nonedge )+1e-4) - 0.2 * torch.sum ( entropy_edge ) / (torch.sum ( mask_edge )+1e-4)
        return loss


class EntropyLoss ( nn.Module ) :
    def __init__(self) :
        super ( EntropyLoss, self ).__init__ ()

    def forward(self, x, mask, mask_label, loss_type='max') :
        # mask_size = mask.size()[1:3]
        # x_softmax = F.softmax(x, dim = 1)
        # x_logsoftmax = F.log_softmax(x, dim = 1)
        # x_softmax_up = F.interpolate(x_softmax, size=mask_size, mode='bilinear', align_corners=True)
        # x_logsoftmax_up = F.interpolate(x_logsoftmax, size=mask_size, mode='bilinear', align_corners=True)
        # b = x_softmax_up * x_logsoftmax_up

        mask = (mask!=mask_label) & (mask!=255)
        if loss_type=='max':
            b = F.softmax ( x, dim=1 ) * F.log_softmax ( x, dim=1 )
            b = -1.0 * torch.sum ( b, dim=1 )
        else:
            b = F.softmax ( x, dim=1 )
            b_max2, _=torch.topk(b,2,dim=1)
            b = torch.mean(torch.abs(b_max2-0.5),dim=1)
        entropy = b.mul ( mask )
        loss = torch.sum ( entropy ) / torch.sum ( mask )
        return loss


class MSELoss_mask ( nn.Module ) :
    def __init__(self) :
        super ( MSELoss_mask, self ).__init__ ()
        self.criterion_mse = nn.MSELoss ( reduction='none' )
        self.criterion_mse_mean = nn.MSELoss ( reduction='mean' )

    def forward(self, x1, x2, mask=None, mask_label=1) :
        mse_loss = self.criterion_mse ( x1, x2 )
        input_size = x1.size ()[2 :4]
        batch_size = x1.size ()[1]
        mask = F.interpolate ( torch.unsqueeze ( mask, 1 ).float (), size=input_size, mode='nearest' )
        mask_ignore = (mask == mask_label)
        mse_mask_loss = mse_loss.mul ( mask_ignore )
        loss = torch.sum ( mse_mask_loss ) / (torch.sum ( mask_ignore ) * batch_size)
        return loss

class EdgeLoss ( nn.Module ) :
    def __init__(self) :
        super ( EdgeLoss, self ).__init__ ()
        
    def forward(self, pred_sg_up, edge_v, adj) :
        pred_seg_softmax = torch.softmax ( pred_sg_up, 1 )
        batch_size = pred_sg_up.size ()[0]
        channel = pred_sg_up.size ()[1]
        adj_size = adj.size()[1]
        pred_mean = torch.zeros(batch_size, channel, 256).cuda(non_blocking=True)
        edge = torch.flatten ( edge_v, start_dim=1 )
        pred_seg = torch.flatten ( pred_seg_softmax, start_dim=2 )
        for i in range ( batch_size ) :
            unique_labels, unique_id, counts = torch.unique ( edge[i], return_inverse=True, return_counts=True )
            num_instances = unique_labels.size ()[0]
            unique_id_repeat = unique_id.unsqueeze ( 0 ).repeat ( channel, 1 )
            segmented_sum = torch.zeros ( channel, num_instances ).cuda ().scatter_add ( dim=1, index=unique_id_repeat,
                                                                                         src=pred_seg[i] )
            mu = torch.div ( segmented_sum, counts )
            pred_mean[i,:,unique_labels] = mu
        #for i in range ( adj_size ) :
        #    for j in range(batch_size):
        #        mask = ( edge_v[j,:,:] == i + 1 )
        #        if torch.sum(mask) !=0 :
        #            mask_repeat = mask.unsqueeze ( 1 ).repeat ( 1, channel, 1, 1 )
        #            pred_i = torch.mul(pred_seg_softmax, mask_repeat)
        #            pred_mean[:,:,i] = torch.sum(pred_i,dim=[2,3]) / (torch.sum(mask_repeat,dim=[2,3])+1e-6)
        pred_mean = pred_mean[:,:,0:-4]
        pred_mean = pred_mean / (torch.norm(pred_mean, dim=1 ,keepdim=True)+1e-6)
        pred_loss = torch.bmm(pred_mean.permute(0,2,1),pred_mean)
        pred_adj_loss = torch.mul(pred_loss, adj)
        loss = torch.mean(torch.sum(pred_adj_loss,dim = 1)/(torch.sum(adj, dim = 1)+1e-6))
        
        return loss
        
class MultiClassLoss ( nn.Module ) :
    def __init__(self, reduction='max', delta=2.3) :
        super ( MultiClassLoss, self ).__init__ ()
        self.reduction = reduction
        self.delta = delta
        
    def forward(self, pred_sg_up) :
        pred_seg_softmax = torch.softmax ( pred_sg_up, 1 )
        if self.reduction=='mean':
            pred_class = torch.mean(pred_seg_softmax, dim=[0,2,3])
        elif self.reduction=='max':
            pred_class = torch.max(pred_seg_softmax, dim=3)[0]
            pred_class = torch.max(pred_class, dim=2)[0]
            pred_class = torch.max(pred_class, dim=0)[0]
            pred_class = pred_class / torch.sum(pred_class)
        
        b = pred_class * torch.log( pred_class )
        loss = -1.0 * torch.sum ( b )
        loss = torch.clamp ( loss , max= self.delta )
        return loss

class EdgeLoss_discriminate ( nn.Module ) :
    def __init__(self, delta=0.1, edge_balance=False) :
        super ( EdgeLoss_discriminate, self ).__init__ ()
        self.edge_balance = edge_balance
        self.delta = delta

    def forward(self, pred_sg_up, edge_v, edge_label) :
        edge = torch.flatten ( edge_v, start_dim=1 )
        pred_seg_softmax = torch.softmax ( pred_sg_up, 1 )
        pred_seg = torch.flatten ( pred_seg_softmax, start_dim=2 )
        batch_size = pred_seg.size ()[0]
        channel = pred_seg.size ()[1]
        var_term = 0.0
        for i in range ( batch_size ) :
            unique_labels, unique_id, counts = torch.unique ( edge[i], return_inverse=True, return_counts=True )
            num_instances = unique_labels.size ()[0]
            unique_id_repeat = unique_id.unsqueeze ( 0 ).repeat ( channel, 1 )
            segmented_sum = torch.zeros ( channel, num_instances ).cuda ().scatter_add ( dim=1, index=unique_id_repeat,
                                                                                         src=pred_seg[i] )
            mu = torch.div ( segmented_sum, counts )
            mu_expand = torch.gather ( mu, 1, unique_id_repeat )
            tmp_distance = pred_seg[i] - mu_expand
            distance = torch.sum ( torch.abs ( tmp_distance ), dim=0 )
            distance = torch.clamp ( distance - self.delta, min=0.0 )
            if self.edge_balance == False :
                mask = (edge[i] != edge_label) & (edge[i] != 255)
                l_var = torch.sum ( distance * mask ) / (torch.sum ( mask ) + 1e-5)
            else :
                l_var = torch.zeros ( num_instances ).cuda ().scatter_add ( dim=0, index=unique_id, src=distance )
                l_var = torch.div ( l_var, counts )
                mask = (unique_labels != 0) & (unique_labels != 255)
                l_var = torch.sum ( l_var * mask ) / (torch.sum ( mask ) + 1e-5)
            var_term = var_term + l_var
        loss_edge = var_term / batch_size
        return loss_edge