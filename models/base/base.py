'''
Function:
    Base segmentor for all supported segmentors
Author:
    Zhenchao Jin
'''
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from utils.losses import BuildLoss
from ..backbones import BuildBackbone, BuildActivation, BuildNormalization, NormalizationBuilder
from .classifier import Classifier2D, Classifier1D

'''BaseSegmentor'''
class BaseSegmentor(nn.Module):
    def __init__(self, cfg, mode):
        super(BaseSegmentor, self).__init__()
        self.cfg = cfg
        self.mode = mode
        assert self.mode in ['train', 'val', 'test']
        # parse align_corners, normalization layer and activation layer cfg
        self.align_corners, self.norm_cfg, self.act_cfg = cfg['align_corners'], cfg['norm_cfg'], cfg['act_cfg']
        # build backbone
        backbone_cfg = copy.deepcopy(cfg['backbone'])
        if 'norm_cfg' not in backbone_cfg:
            backbone_cfg.update({'norm_cfg': copy.deepcopy(self.norm_cfg)})
        self.backbone_net = BuildBackbone(backbone_cfg)
        # build classifier
        self.classifer = Classifier2D(cfg['classifier']['last_inchannels'], cfg['num_classes'], cfg['classifier']['dropout'])
    '''forward'''
    def forward(self, x, targets=None):
        raise NotImplementedError('not to be implemented')
    '''customizepredsandlosses'''
    def customizepredsandlosses(self, predictions_image, predictions_mask, targets, losses_cfg, img_size, auto_calc_loss=True, map_preds_to_tgts_dict=None):
        predictions_mask = F.interpolate(predictions_mask, size=img_size, mode='bilinear', align_corners=self.align_corners)
        outputs_dict = {'pix_cls': predictions_mask}
        outputs_dict['img_cls'] = predictions_image
        if not auto_calc_loss: return outputs_dict
        return self.calculatelosses(predictions=outputs_dict, targets=targets, losses_cfg=losses_cfg)
    '''inference'''
    def inference(self, images, forward_args=None):
        # assert and initialize
        images = images.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
        # inference
        if forward_args is None: img_outputs, pix_outputs = self(images)
        else: img_outputs, pix_outputs = self(images, **forward_args)
        # return outputs
        return img_outputs, pix_outputs
    '''transforminputs'''
    def transforminputs(self, x_list, selected_indices=None):
        if selected_indices is None:
            if self.cfg['backbone']['type'] in ['HRNet']:
                selected_indices = (0, 0, 0, 0)
            else:
                selected_indices = (0, 1, 2, 3)
        outs = []
        for idx in selected_indices:
            outs.append(x_list[idx])
        return outs
    '''freezenormalization'''
    def freezenormalization(self, norm_list=None):
        if norm_list is None:
            norm_list=(nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
        for module in self.modules():
            if NormalizationBuilder.isnorm(module, norm_list):
                module.eval()
                for p in module.parameters():
                    p.requires_grad = False
    '''calculatelosses'''
    def calculatelosses(self, predictions, targets, losses_cfg):
        # parse targets
        target_seg = targets['seg_target']
        target_img = targets['label']
        assert len(predictions) == len(losses_cfg), 'length of losses_cfg should be equal to the one of predictions'
        # calculate loss according to losses_cfg
        losses_log_dict = {}
        for loss_name, loss_cfg in losses_cfg.items():
            if 'pix' in loss_name:
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name], target=target_seg, loss_cfg=loss_cfg,
                )
            else:
                losses_log_dict[loss_name] = self.calculateloss(
                    prediction=predictions[loss_name], target=target_img, loss_cfg=loss_cfg,
                )
        # summarize and convert losses_log_dict
        loss = 0
        for loss_key, loss_value in losses_log_dict.items():
            loss_value = loss_value.mean()
            loss = loss + loss_value
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses_log_dict[loss_key] = loss_value.item()
        losses_log_dict.update({'loss_total': sum(losses_log_dict.values())})
        # return the loss and losses_log_dict
        return loss, losses_log_dict
    '''calculateloss'''
    def calculateloss(self, prediction, target, loss_cfg):
        assert isinstance(loss_cfg, (dict, list))
        # calculate the loss, dict means single-type loss and list represents multiple-type losses
        if isinstance(loss_cfg, dict):
            loss = BuildLoss(loss_cfg)(prediction, target)
        else:
            loss = 0
            for l_cfg in loss_cfg:
                loss = loss + BuildLoss(l_cfg)(prediction, target)
        # return the loss
        return loss