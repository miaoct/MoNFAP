'''fcn_resnet50os8_ade20k'''
import copy
from .base_cfg import SEGMENTOR_CFG
from .._base_ import DATASET_CFG_OpenForensics

# deepcopy
SEGMENTOR_CFG = copy.deepcopy(SEGMENTOR_CFG)
# modify dataset config
SEGMENTOR_CFG['dataset'] = DATASET_CFG_OpenForensics.copy()
# modify dataloader config
SEGMENTOR_CFG['dataloader'] = {
    'train': {
        'batch_size_per_gpu': 32, 'num_workers_per_gpu': 2, 'shuffle': True, 'pin_memory': True, 'drop_last': True,
    },
    'val': {
        'batch_size_per_gpu': 1, 'num_workers_per_gpu': 2, 'shuffle': False, 'pin_memory': True, 'drop_last': False,
    },
    'test': {
        'batch_size_per_gpu': 1, 'num_workers_per_gpu': 2, 'shuffle': False, 'pin_memory': True, 'drop_last': False,
    }
}
# modify scheduler config
SEGMENTOR_CFG['scheduler'] = {
        'type': 'PolyScheduler', 'max_epochs': 40, 'power': 0.9,
        'optimizer': {
            'type': 'AdamW', 'lr': 0.00006, 'betas': (0.9, 0.999), 'weight_decay': 0.01, 'params_rules': {},
        }
    }
# modify loss config
SEGMENTOR_CFG['losses'] = {
        'pix_prompt': {
            'pix_prompt_fake':{'type': 'CrossEntropyLoss', 'scale_factor': 10, 'ignore_index': 255, 'reduction': 'mean'},
            'pix_prompt_real':{'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
            },
        'pix_cls': {
            'pix_cls_fake':{'type': 'CrossEntropyLoss', 'scale_factor': 10, 'ignore_index': 255, 'reduction': 'mean'},
            'pix_cls_real':{'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
            },
        'img_cls': {'type': 'CrossEntropyLoss', 'scale_factor': 1.0, 'ignore_index': 255, 'reduction': 'mean'},
    }
# modify other segmentor configs
SEGMENTOR_CFG['type'] = 'Forgery2Mask'
SEGMENTOR_CFG['num_classes'] = 2
SEGMENTOR_CFG['backbone'] = {
    'type': 'ConvNeXtV2', 'structure_type': 'convnextv2_atto_1k_224_ema', 'arch': 'atto', 'pretrained': True, 'drop_path_rate': 0.4,
    'out_indices':(0,1,2,3),'selected_indices': (0,1,2,3), 'norm_cfg': {'type': 'LayerNormConvNeXtV2', 'eps': 1e-6}, 'act_cfg': {'type': 'GELU'},
}
# modify maskdecoder config
SEGMENTOR_CFG['mask_decoder'] = {
        'moe_experts': 4, 'moe_topk': 4, 'noise_dim': [40, 80, 160, 320], 'transformer_dim': [320, 160, 80, 40], 'attn_mask_thr': 0.5, 'cls_dropout': 0.1,
    }
# modify aux prefiction head config
SEGMENTOR_CFG['pre_head'] = {
        'in_channels': 320, 'feats_channels': 512,
    }

SEGMENTOR_CFG['output_dir'] = './output/models/MoNFAP'
SEGMENTOR_CFG['work_dir'] = 'Forgery2Mask_convnextv2_atto_OpenForensics'
SEGMENTOR_CFG['logfilepath'] = 'Forgery2Mask_convnextv2_OpenForensics'
SEGMENTOR_CFG['resultsavepath'] = 'Forgery2Mask_convnextv2_OpenForensics_results.pkl'
SEGMENTOR_CFG['log_interval_iterations'] = 1000
SEGMENTOR_CFG['eval_interval_epochs'] = 1
SEGMENTOR_CFG['save_interval_epochs'] = 1
