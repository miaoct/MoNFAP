import copy
from utils import BaseModuleBuilder
from .faceforensics import FFIWDataset, OFDataset, MFDataset, FFDataset, FFIWAugDataset, OFAugDataset


'''DatasetBuilder'''
class DatasetBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'Pixel_FFIW10K': FFIWDataset, 
        'Pixel_OpenForensics': OFDataset, 
        'Pixel_FFIW10K_Aug': FFIWAugDataset, 
        'Pixel_OpenForensics_Aug': OFAugDataset, 
        'Pixel_ManualFake': MFDataset, 
        'Pixel_FF++': FFDataset,
    }
    '''build'''
    def build(self, mode, logger_handle, dataset_cfg):
        dataset_cfg = copy.deepcopy(dataset_cfg)
        train_cfg, val_cfg, test_cfg = dataset_cfg.pop('train', {}), dataset_cfg.pop('val', {}), dataset_cfg.pop('test', {})
        if mode == 'train':
            dataset_cfg.update(train_cfg)
        elif mode == 'val':
            dataset_cfg.update(val_cfg)
        else:
            dataset_cfg.update(test_cfg)
        module_cfg = {
            'mode': mode, 'logger_handle': logger_handle, 'dataset_cfg': dataset_cfg, 'type': dataset_cfg['type'],
        }
        return super().build(module_cfg)


'''BuildDataset'''
BuildDataset = DatasetBuilder().build