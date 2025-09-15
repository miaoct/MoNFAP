'''
Function:
    Implementation of SigmoidFocalLoss
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from mmcv.ops import sigmoid_focal_loss
except:
    sigmoid_focal_loss = None


'''SigmoidFocalLoss'''
class SigmoidFocalLoss(nn.Module):
    def __init__(self, scale_factor=1.0, gamma=2, alpha=0.25, weight=None, reduction='mean', ignore_index=None, lowest_loss_value=None):
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.scale_factor = scale_factor
        self.lowest_loss_value = lowest_loss_value
    '''forward'''
    def forward(self, prediction, target):
        # fetch attributes
        alpha, gamma, weight, lowest_loss_value = self.alpha, self.gamma, self.weight, self.lowest_loss_value
        scale_factor, reduction, ignore_index = self.scale_factor, self.reduction, self.ignore_index
        # filter according to ignore_index
        if ignore_index is not None:
            nums = prediction.size(1)
            mask = (target != ignore_index)
            prediction = prediction.squeeze(1)
        
            prediction, target = prediction[mask].view(-1,1), target[mask].view(-1,1)
    
        # calculate loss
        loss = self.calculate_sigmoid_focal_loss(prediction, target, nums, gamma, alpha)
        loss = loss * scale_factor
        if lowest_loss_value is not None:
            loss = torch.abs(loss - lowest_loss_value) + lowest_loss_value
        # return
        return loss
    
    def calculate_sigmoid_focal_loss(self, inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_masks