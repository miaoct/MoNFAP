'''
Function:
    Implementation of BaseDataset
Author:
    Zhenchao Jin
'''
import os
import cv2
import torch
import numpy as np
import collections
from PIL import Image
from .transforms import BuildDataTransform, DataTransformBuilder, Compose


'''BaseDataset'''
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, logger_handle, dataset_cfg):
        # assert
        assert mode in ['train', 'val', 'test', 'test_vis']
        # set attributes
        self.mode = mode
        # self.ann_ext = '.png'
        # self.image_ext = '.jpg'
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        self.repeat_times = dataset_cfg.get('repeat_times', 1)
        self.transforms = self.constructtransforms(self.dataset_cfg['data_pipelines'])
    '''getitem'''
    def __getitem__(self, idx):
        im_name, maskpath, img_label = self.images[idx]
        # read sample_meta
        sample_meta = self.read(im_name, maskpath)
        sample_meta.update({"img_target":img_label})
        if self.dataset_cfg['type'] == 'Pixel_OpenForensics':
            sample_meta.update({"id": '_'.join(im_name.split('/')[-3:])})
        else:
            sample_meta.update({"id": '_'.join(im_name.split('/')[-4:])})
        # synctransforms
        # sample_meta['seg_target'][sample_meta['seg_target'] == 255] = 1.
        # if self.mode == 'test' or self.mode == 'val': sample_meta['seg_gt'][sample_meta['seg_gt'] == 255] = 1.
        sample_meta = self.synctransforms(sample_meta)
        # return
        return sample_meta
    '''len'''
    def __len__(self):
        return len(self.images) * self.repeat_times
    '''read sample_meta'''
    def read(self, imagepath, annpath):
        # read image
        image = cv2.imread(imagepath)
        # read annotation
        if os.path.exists(annpath):
            seg_target = cv2.imread(annpath, cv2.IMREAD_GRAYSCALE)
        else:
            seg_target = np.zeros((image.shape[0], image.shape[1]))
        # auto transform seg_target to train labels
        if hasattr(self, 'clsid2label'):
            for key, value in self.clsid2label.items():
                seg_target[seg_target == key] = value
        # construct sample_meta
        sample_meta = {
            'image': image, 'seg_target': seg_target, 'width': seg_target.shape[1], 'height': seg_target.shape[0],
        }
        if self.mode == 'test' or self.mode == 'val': sample_meta.update({'seg_gt': seg_target.copy()})
        # sample_meta.update({'seg_gt': seg_target.copy()})
        # return
        return sample_meta
    '''constructtransforms'''
    def constructtransforms(self, data_pipelines):
        # supported transforms
        transforms = []
        for data_pipeline in data_pipelines:
            if isinstance(data_pipeline, collections.abc.Sequence):
                assert len(data_pipeline) == 2
                assert isinstance(data_pipeline[1], dict)
                transform_type, transform_cfg = data_pipeline
                transform_cfg['type'] = transform_type
                transform = BuildDataTransform(transform_cfg)
            else:
                assert isinstance(data_pipeline, dict)
                transform = BuildDataTransform(data_pipeline)
            transforms.append(transform)
        transforms = Compose(transforms)
        # return
        return transforms
    '''synctransforms'''
    def synctransforms(self, sample_meta):
        if self.mode in ['test','val', 'test_vis']:
            seg_target = sample_meta.pop('seg_target')
        sample_meta = self.transforms(sample_meta)
        if self.mode in ['test','val', 'test_vis']:
            sample_meta['seg_target'] = seg_target
        return sample_meta
    '''randompalette'''
    @staticmethod
    def randompalette(num_classes):
        palette = [0] * (num_classes * 3)
        for j in range(0, num_classes):
            i, lab = 0, j
            palette[j * 3 + 0], palette[j * 3 + 1], palette[j * 3 + 2] = 0, 0, 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        palette = np.array(palette).reshape(-1, 3)
        palette = palette.tolist()
        return palette