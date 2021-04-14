import warnings
import numpy as np
from torch import nn

from torch.nn import functional as F

import torch
import math
from torch import Tensor, einsum
from torch.nn.modules.loss import _Loss


def get_soft_label(input_tensor, num_class, data_type = 'float'):
    """
        convert a label tensor to soft label 
        input_tensor: tensor with shae [B, 1, D, H, W]
        output_tensor: shape [B, num_class, D, H, W]
    """
    tensor_list = []
    for i in range(num_class):
        temp_prob = input_tensor == i*torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim = 1)
    if(data_type == 'float'):
        output_tensor = output_tensor.float()
    elif(data_type == 'double'):
        output_tensor = output_tensor.double()
    else:
        raise ValueError("data type can only be float and double: {0:}".format(data_type))

    return output_tensor

def reshape_tensor_to_2D(x):
    """
    reshape input variables of shape [B, C, D, H, W] to [voxel_n, C]
    """
    tensor_dim = len(x.size())
    num_class  = list(x.size())[1]
    if(tensor_dim == 5):
        x_perm  = x.permute(0, 2, 3, 4, 1)
    elif(tensor_dim == 4):
        x_perm  = x.permute(0, 2, 3, 1)
    else:
        raise ValueError("{0:}D tensor not supported".format(tensor_dim))
    
    y = torch.reshape(x_perm, (-1, num_class)) 
    return y 



class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num):
        super(CrossEntropyLoss, self).__init__()
        self.cls_num = cls_num
    def forward(self, predict, soft_y):

        predict = nn.Softmax(dim = 1)(predict)
        soft_y = get_soft_label(soft_y.unsqueeze(1), self.cls_num).cuda()

        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)

        ce = - soft_y * torch.log(predict)
        ce = torch.sum(ce, dim = 1)
        ce = torch.mean(ce)  
        return ce

class WeightedCELoss(nn.Module):
    def __init__(self, cls_num, pix_weights):
        super().__init__()
        self.cls_num = cls_num
        self.pix_weights = pix_weights
    def forward(self, predict, soft_y):
        predict = nn.Softmax(dim = 1)(predict)
        soft_y = get_soft_label(soft_y.unsqueeze(1), self.cls_num).cuda()

        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y)

        ce = - soft_y * torch.log(predict)
        ce = torch.sum(ce, dim = 1)

        pix_w = torch.ones_like(ce).cuda() * soft_y[:,0] * self.pix_weights[0] + \
            torch.ones_like(ce).cuda() * soft_y[:,1] * self.pix_weights[1]
        ce    = torch.sum(ce * pix_w) / torch.sum(pix_w)

        return ce


def get_classwise_dice(predict, soft_y, pix_w = None):
    """
    get dice scores for each class in predict (after softmax) and soft_y
    """
    
    if(pix_w is None):
        y_vol = torch.sum(soft_y,  dim = 0)
        p_vol = torch.sum(predict, dim = 0)
        intersect = torch.sum(soft_y * predict, dim = 0)
    else:
        y_vol = torch.sum(soft_y * pix_w,  dim = 0)
        p_vol = torch.sum(predict * pix_w, dim = 0)
        intersect = torch.sum(soft_y * predict * pix_w, dim = 0)
    dice_score = (2.0 * intersect + 1e-5)/ (y_vol + p_vol + 1e-5)
    return dice_score 

class DiceLoss(nn.Module):
    def __init__(self, cls_num):
        super(DiceLoss, self).__init__()
        self.cls_num = cls_num

    def forward(self, predict, soft_y):
        predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y = get_soft_label(soft_y.unsqueeze(1), self.cls_num).cuda()
        soft_y  = reshape_tensor_to_2D(soft_y) 

        dice_score = get_classwise_dice(predict, soft_y, pix_w=None)
        avg_dice = torch.mean(dice_score)   
        dice_loss  = 1.0 - avg_dice
        return dice_loss

class WeightedDiceLoss(nn.Module):
    def __init__(self, cls_num, cls_weights):
        super().__init__()
        self.cls_num = cls_num
        self.cls_weights = cls_weights
    
    def forward(self, predict, soft_y):
        predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y = get_soft_label(soft_y.unsqueeze(1), self.cls_num).cuda()
        soft_y  = reshape_tensor_to_2D(soft_y) 

        dice_score = get_classwise_dice(predict, soft_y, pix_w=None)
        weighted_dice = dice_score * torch.tensor(self.cls_weights).cuda()
        avg_dice =  weighted_dice.sum() / torch.tensor(self.cls_weights).cuda().sum()

        dice_loss  = 1.0 - avg_dice
        return dice_loss


class DiceWithCrossEntropyLoss(nn.Module):
    def __init__(self, cls_num, ce_weight=1.0):
        super(DiceWithCrossEntropyLoss, self).__init__()

        self.cls_num = cls_num
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(cls_num)
        self.ce_loss   = CrossEntropyLoss(cls_num)

    def forward(self, predict, soft_y):
        loss1 = self.dice_loss(predict, soft_y)
        loss2 = self.ce_loss(predict, soft_y)
        loss = loss1 + self.ce_weight * loss2
        return loss 
        
class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, cls_num, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.cls_num = cls_num
    def forward(self, pred, soft_y):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        one_hot_gt = get_soft_label(soft_y.unsqueeze(1), self.cls_num).cuda()

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss


class BDDiceLoss(nn.Module):
    def __init__(self, cls_num, theta0=3, theta=5):
        super().__init__()
        self.cls_num = cls_num
        self.theta0 = theta0
        self.theta = theta
        self.dice = DiceLoss(cls_num=self.cls_num)
        self.bdl = BoundaryLoss(cls_num=self.cls_num)

    def forward(self, pred, soft_y):

        bdloss = self.bdl(pred, soft_y)
        dice_loss = self.dice(pred, soft_y)

        return bdloss + dice_loss

class NoiseRobustDiceLoss(nn.Module):
    """
    Noise-robust Dice loss according to the following paper. 
        G. Wang et al. A Noise-Robust Framework for Automatic Segmentation of COVID-19 
        Pneumonia Lesions From CT Images, IEEE TMI, 2020.
    """
    def __init__(self, gamma, cls_num):
        super(NoiseRobustDiceLoss, self).__init__()
        self.gamma = gamma
        self.cls_num = cls_num
    def forward(self, predict, soft_y):
        predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y = get_soft_label(soft_y.unsqueeze(1), self.cls_num).cuda()
        soft_y  = reshape_tensor_to_2D(soft_y) 

        numerator = torch.abs(predict - soft_y)
        numerator = torch.pow(numerator, self.gamma)
        denominator = predict + soft_y 
        numer_sum = torch.sum(numerator,  dim = 0)
        denom_sum = torch.sum(denominator,  dim = 0)
        loss_vector = numer_sum / (denom_sum + 1e-5)

        loss = torch.mean(loss_vector)   
        return loss


class ExpLogLoss(nn.Module):
    """
    The exponential logarithmic loss in this paper: 
        K. Wong et al.: 3D Segmentation with Exponential Logarithmic Loss for Highly 
        Unbalanced Object Sizes. MICCAI 2018.
    """
    def __init__(self, cls_num, w_dice=0.5, gamma=0.3):
        super(ExpLogLoss, self).__init__()
        self.cls_num = cls_num
        self.w_dice = w_dice
        self.gamma  = gamma

    def forward(self, predict, soft_y):

        predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y = get_soft_label(soft_y.unsqueeze(1), self.cls_num).cuda()
        soft_y  = reshape_tensor_to_2D(soft_y)

        dice_score = get_classwise_dice(predict, soft_y)
        dice_score = 0.01 + dice_score * 0.98
        exp_dice   = -torch.log(dice_score)
        exp_dice   = torch.pow(exp_dice, self.gamma)
        exp_dice   = torch.mean(exp_dice)

        predict= 0.01 + predict * 0.98
        wc     = torch.mean(soft_y, dim = 0)
        wc     = 1.0 / (wc + 0.1)
        wc     = torch.pow(wc, 0.5)
        ce     = - torch.log(predict)
        exp_ce = wc * torch.pow(ce, self.gamma)
        exp_ce = torch.sum(soft_y * exp_ce, dim = 1)
        exp_ce = torch.mean(exp_ce)

        loss = exp_dice * self.w_dice + exp_ce * (1.0 - self.w_dice)
        return loss


class LACE(nn.Module):
    def __init__(self, priors, tau = 1.0, cls_num = 2):
        super().__init__()
        self.priors = priors
        self.tau = tau
        self.cls_num = cls_num
    def forward(self, predict, soft_y):

        predict = reshape_tensor_to_2D(predict)
        log_prior = torch.tensor(np.log(np.array(self.priors) + 1e-8)).cuda()
        predict = predict + self.tau * log_prior.unsqueeze(0)

        predict = nn.Softmax(dim = 1)(predict)
        soft_y = get_soft_label(soft_y.unsqueeze(1), self.cls_num).cuda()

        soft_y  = reshape_tensor_to_2D(soft_y)

        ce = - soft_y * torch.log(predict)
        ce = torch.sum(ce, dim = 1)
        ce = torch.mean(ce)  

        return ce

class BD_LACE_WDiceLoss(nn.Module):
    def __init__(self, cls_num, priors, dice_cls_weights, tau = 1.0):
        super().__init__()
        self.cls_num = cls_num

        self.WeightedDiceLoss = WeightedDiceLoss(cls_num=self.cls_num, cls_weights=dice_cls_weights)
        self.LACE = LACE(priors=priors, tau = tau, cls_num = self.cls_num)
        self.bdl = BoundaryLoss(cls_num=self.cls_num)

    def forward(self, predict, soft_y):

        wdice_l = self.WeightedDiceLoss(predict, soft_y)
        lace_l = self.LACE(predict, soft_y)
        bd_l = self.bdl(predict, soft_y)

        return wdice_l + lace_l + bd_l

