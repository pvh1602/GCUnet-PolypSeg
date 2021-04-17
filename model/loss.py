import torch
import torch.nn as nn
import torch.nn.functional as F



# STRUCTURE LOSS
def structure_loss(pred, mask):
    # print("Using structure loss!")
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def BCE_loss(pred, mask):
    return F.binary_cross_entropy_with_logits(pred, mask)


def MSE_loss(pred, mask):
    return F.mse_loss(pred, mask)


def Dice_loss(pred, mask):
    smooth = 1.
    intersection = (mask * pred).sum()  
    score = (2. * intersection + smooth) / (mask.sum() + pred.sum() + smooth) 
    return 1 - score



