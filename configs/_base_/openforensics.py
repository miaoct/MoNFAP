'''OpenForensics_1024x1024'''
import os


'''DATASET_CFG_OpenForensics_1024x1024'''
DATASET_CFG_OpenForensics = {
    'type': 'Pixel_OpenForensics',
    'rootdir': './data/preprocessed/',
    'real_training': True,
    'train': {
        'data_pipelines': [
            ('Resize', {'output_size': (1024, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'flip_prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
        ],
    },
    'val': {
        'data_pipelines': [
            ('Resize', {'output_size': (512, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    },
    'test': {
        'data_pipelines': [
            ('Resize', {'output_size': (512, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    }
}

'''DATASET_CFG_OpenForensicsAug_1024x1024'''
DATASET_CFG_OpenForensics_Aug = {
    'type': 'Pixel_OpenForensics_Aug',
    'rootdir': './data/preprocessed/',
    'real_training': True,
    'train': {
        'data_pipelines': [
            ('Resize', {'output_size': (1024, 512), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
            ('RandomCrop', {'crop_size': (512, 512), 'one_category_max_ratio': 0.75}),
            ('RandomFlip', {'flip_prob': 0.5}),
            ('PhotoMetricDistortion', {}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
            ('Padding', {'output_size': (512, 512), 'data_type': 'tensor'}),
        ],
    },
    'val': {
        'data_pipelines': [
            ('Resize', {'output_size': (512, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    },
    'test': {
        'data_pipelines': [
            ('Resize', {'output_size': (512, 512), 'keep_ratio': True, 'scale_range': None}),
            ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
            ('ToTensor', {}),
        ],
    }
}