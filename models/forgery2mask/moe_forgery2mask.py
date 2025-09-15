'''
Function:
    Implementation of foregry2mask
Author:
    Changtao Miao
'''
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from ..base import BaseSegmentor, SimpleGate, NoiseMoELayer
from ..backbones import BuildActivation, BuildNormalization
from .masked_decoder import MaskDecoder

'''mask decoder'''
class Forgery2Mask(BaseSegmentor):
    def __init__(self, cfg, mode):
        super(Forgery2Mask, self).__init__(cfg, mode)
        norm_cfg, act_cfg, head_cfg = self.norm_cfg, self.act_cfg, cfg['pre_head']
        # build noise prompt layers
        self.dims = cfg['mask_decoder']['noise_dim']
        self.noise_layers = nn.ModuleList()
        for i in range(4):
            moe = NoiseMoELayer(SimpleGate(in_channels=self.dims[i], num_experts=cfg['mask_decoder']['moe_experts'], top_k=cfg['mask_decoder']['moe_topk']),
                                channels=self.dims[i],
                                num_experts=cfg['mask_decoder']['moe_experts'], 
                                top_k=cfg['mask_decoder']['moe_topk'])
            self.noise_layers.append(moe)
        # build decoder
        self.decoder = MaskDecoder(transformer_dim=cfg['mask_decoder']['transformer_dim'],
                                   num_classes=cfg['num_classes'],
                                   attn_mask_thr=cfg['mask_decoder']['attn_mask_thr'],
                                   cls_dropout=cfg['mask_decoder']['cls_dropout'],
                                   )
        # build prediction head
        self.fcn = nn.Sequential(
                nn.Conv2d(head_cfg['in_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
                nn.Conv2d(head_cfg['feats_channels'], head_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(placeholder=head_cfg['feats_channels'], norm_cfg=norm_cfg),
                BuildActivation(act_cfg),
                nn.Conv2d(head_cfg['feats_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
            )
        
        del self.classifer
        
    '''forward feature'''
    def prompt_feature(self, img_outputs):
        prompt_outs, loss_outs = [], []
        for i, noise in enumerate(self.noise_layers):
            
            outputs = noise(img_outputs[i])
            loss_outs.append(outputs['aux_loss'])
            prompt_outs.append(outputs['output'])
        
        return prompt_outs, loss_outs
    
    '''forward'''
    def forward(self, image, targets=None):
        img_size = image.size(2), image.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(image), selected_indices=self.cfg['backbone'].get('selected_indices'))
        noise_prompts, moe_losses = self.prompt_feature(backbone_outputs)
        prompt_mask_outs = self.fcn(backbone_outputs[-1])
        # feed to decoder
        predictions_mask, predictions_image = [], []
        for i in range(image.size(0)):
            curr_embeddings = backbone_outputs[3][i].unsqueeze(0)
                               
            dense_embeddings = [noise_prompts[3][i].unsqueeze(0), 
                               noise_prompts[2][i].unsqueeze(0),
                               noise_prompts[1][i].unsqueeze(0),
                               noise_prompts[0][i].unsqueeze(0)
                               ]
            attn_mask = prompt_mask_outs[i].unsqueeze(0)
            pred_mask, pred_image = self.decoder(curr_embeddings, dense_embeddings, attn_mask)
            
            predictions_mask.append(pred_mask)
            predictions_image.append(pred_image)
            
        predictions_mask = torch.cat(predictions_mask, dim=0)
        predictions_image = torch.cat(predictions_image, dim=0)
        # forward according to the mode
        if self.mode == 'train':
            img_cls = self.calculateloss(prediction=predictions_image, target=targets['label'], loss_cfg=self.cfg['losses']['img_cls'],)
            
            predictions_mask = F.interpolate(predictions_mask, size=img_size, mode='bilinear', align_corners=self.align_corners)
            pix_cls_real, pix_cls_fake = self.calculateloss_rf(predictions=predictions_mask, targets=targets, losses_cfg=self.cfg['losses']['pix_cls'],)
            
            prompt_mask = F.interpolate(prompt_mask_outs, size=img_size, mode='bilinear', align_corners=self.align_corners)
            prompt_loss_real, prompt_loss_fake = self.calculateloss_rf(predictions=prompt_mask, targets=targets, losses_cfg=self.cfg['losses']['pix_prompt'])
            
            losses_dict = {'img_cls': img_cls, 'pix_cls_real': pix_cls_real, 'pix_cls_fake': pix_cls_fake, 'prompt_loss_real': prompt_loss_real, 'prompt_loss_fake': prompt_loss_fake,
                           'moe_loss_0': moe_losses[0], 'moe_loss_1': moe_losses[1], 'moe_loss_2': moe_losses[2], 'moe_loss_3': moe_losses[3]}
            loss, losses_log_dict = 0, {}
            for key, value in losses_dict.items():
                loss += value
                value = value.data.clone()
                dist.all_reduce(value.div_(dist.get_world_size()))
                losses_log_dict[key] = value.item()
            losses_log_dict['total'] = sum(losses_log_dict.values())
            
            return loss, losses_log_dict
        return predictions_image, predictions_mask
    
    
    def calculateloss_rf(self, predictions, targets, losses_cfg):
        # parse targets
        target_seg = targets['seg_target']
        target_img = targets['label']
        target_seg_real, target_seg_fake = [], []
        prediction_seg_real, prediction_seg_fake = [], []
        for idx, img_label in enumerate(target_img):
            if img_label == 1:
                target_seg_fake.append(target_seg[idx].unsqueeze(0))
                prediction_seg_fake.append(predictions[idx].unsqueeze(0))
            else:
                target_seg_real.append(target_seg[idx].unsqueeze(0))
                prediction_seg_real.append(predictions[idx].unsqueeze(0))
        if len(target_seg_real) != 0:
            target_seg_real = torch.cat(target_seg_real, dim=0)
            prediction_seg_real = torch.cat(prediction_seg_real, dim=0)
        if len(target_seg_fake) != 0:
            target_seg_fake = torch.cat(target_seg_fake, dim=0)
            prediction_seg_fake = torch.cat(prediction_seg_fake, dim=0)
        loss_real, loss_fake = torch.tensor(0.0).type(torch.cuda.FloatTensor), torch.tensor(0.0).type(torch.cuda.FloatTensor)
        for loss_name, loss_cfg in losses_cfg.items():
            if ('real' in loss_name) and (len(target_seg_real) != 0):
                loss_real = self.calculateloss(
                    prediction=prediction_seg_real, target=target_seg_real, loss_cfg=loss_cfg,
                )
            elif ('fake' in loss_name) and (len(target_seg_fake) != 0):
                loss_fake = self.calculateloss(
                    prediction=prediction_seg_fake, target=target_seg_fake, loss_cfg=loss_cfg,
                )
            else:
                # print(len(target_seg_real))
                # print(len(target_seg_fake))
                # print('loss type error!')
                pass
        # return the loss
        return loss_real, loss_fake