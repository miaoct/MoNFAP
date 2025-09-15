import os
import glob
from .base import BaseDataset


class FFIWDataset(BaseDataset):
    num_classes = 2
    classnames = ['real', 'fake']
    palette = [(0, 0, 0), (255, 255, 255)]
    clsid2label = {255: 1}
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(FFIWDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)

        self.datatype = mode
        self.image_dir = os.path.join(dataset_cfg['rootdir'], dataset_cfg['type'])

        if self.datatype == 'train':
            _split_f = os.path.join(self.image_dir, 'train_v2.txt')
        elif self.datatype == 'val':
             _split_f = os.path.join(self.image_dir, 'valid_v2.txt')
        elif self.datatype == 'test':
            _split_f = os.path.join(self.image_dir, 'test_v2.txt')
        elif self.datatype == 'test_vis':
            _split_f = os.path.join(self.image_dir, 'test_vis.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        # obatin images
        self.images = []
        with open(os.path.join(_split_f), "r") as lines:
          for line in lines:
            line = line.rstrip()
            words = line.split()
            img_path = os.path.join(self.image_dir, words[0])

            assert os.path.isfile(img_path), 'file %s error' % img_path
            mask_path = img_path.replace('Image', 'Mask')
            assert os.path.isfile(mask_path), 'file %s error' % mask_path

            if not dataset_cfg['real_training']:
                if (int(words[1])==1):
                    self.images.append((img_path, mask_path, int(words[1])))
            else:
                self.images.append((img_path, mask_path, int(words[1])))
                
        # self.images = self.images[:100]

class FFIWAugDataset(BaseDataset):
    num_classes = 2
    classnames = ['real', 'fake']
    palette = [(0, 0, 0), (255, 255, 255)]
    clsid2label = {255: 1}
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(FFIWAugDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)

        self.datatype = mode
        self.image_dir = os.path.join(dataset_cfg['rootdir'], 'Pixel_FFIW10K')

        if self.datatype == 'test':
            _split_f = os.path.join(self.image_dir, 'test_augv1.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        # obatin images
        self.images = []
        with open(os.path.join(_split_f), "r") as lines:
          for line in lines:
            line = line.rstrip()
            words = line.split()
            img_path = os.path.join(self.image_dir, words[0])

            assert os.path.isfile(img_path), 'file %s error' % img_path
            mask_path = img_path.replace('Aug_Image_v1', 'Mask')
            mask_path = mask_path.split('.')[0] + '.png'
            assert os.path.isfile(mask_path), 'file %s error' % mask_path

            if not dataset_cfg['real_training']:
                if (int(words[1])==1):
                    self.images.append((img_path, mask_path, int(words[1])))
            else:
                self.images.append((img_path, mask_path, int(words[1])))


class OFDataset(BaseDataset):
    num_classes = 2
    classnames = ['real', 'fake']
    palette = [(0, 0, 0), (255, 255, 255)]
    clsid2label = {255: 1}
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(OFDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)

        self.datatype = mode
        self.image_dir = os.path.join(dataset_cfg['rootdir'], dataset_cfg['type'])

        if self.datatype == 'train':
            _split_f = os.path.join(self.image_dir, 'train.txt')
        elif self.datatype == 'val':
             _split_f = os.path.join(self.image_dir, 'valid.txt')
        elif self.datatype == 'test':
            _split_f = os.path.join(self.image_dir, 'test.txt')
        elif self.datatype == 'test_vis':
            _split_f = os.path.join(self.image_dir, 'test_vis.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        # obatin images
        self.images = []
        with open(os.path.join(_split_f), "r") as lines:
          for line in lines:
            line = line.rstrip()
            words = line.split()
            img_path = os.path.join(self.image_dir, words[0])

            assert os.path.isfile(img_path), 'file %s error' % img_path
            mask_path = img_path.replace('Image', 'Mask')
            mask_path = mask_path.split('.')[0] + '.png'
            assert os.path.isfile(mask_path), 'file %s error' % mask_path

            if not dataset_cfg['real_training']:
                if (int(words[1])==1):
                    self.images.append((img_path, mask_path, int(words[1])))
            else:
                self.images.append((img_path, mask_path, int(words[1])))


class OFAugDataset(BaseDataset):
    num_classes = 2
    classnames = ['real', 'fake']
    palette = [(0, 0, 0), (255, 255, 255)]
    clsid2label = {255: 1}
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(OFAugDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)

        self.datatype = mode
        self.image_dir = os.path.join(dataset_cfg['rootdir'], 'Pixel_OpenForensics')

        if self.datatype == 'test':
            _split_f = os.path.join(self.image_dir, 'test_augv1.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        # obatin images
        self.images = []
        with open(os.path.join(_split_f), "r") as lines:
          for line in lines:
            line = line.rstrip()
            words = line.split()
            img_path = os.path.join(self.image_dir, words[0])

            assert os.path.isfile(img_path), 'file %s error' % img_path
            mask_path = img_path.replace('Aug_Image_v1', 'Mask')
            mask_path = mask_path.split('.')[0] + '.png'
            assert os.path.isfile(mask_path), 'file %s error' % mask_path

            if not dataset_cfg['real_training']:
                if (int(words[1])==1):
                    self.images.append((img_path, mask_path, int(words[1])))
            else:
                self.images.append((img_path, mask_path, int(words[1])))


class MFDataset(BaseDataset):
    num_classes = 2
    classnames = ['real', 'fake']
    palette = [(0, 0, 0), (255, 255, 255)]
    clsid2label = {255: 1}
    assert num_classes == len(classnames)
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(MFDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)

        self.datatype = mode
        rootdir = dataset_cfg['rootdir']
        data_name = dataset_cfg['type']
        osn_list = dataset_cfg['osn_list']
        self.image_dir = os.path.join(rootdir, data_name)
        self.images = []
        if self.datatype == 'test':
            for osn in osn_list:
              _split_f = os.path.join(self.image_dir, 'test_'+ str(osn) +'.txt')
              # obatin images
              with open(os.path.join(_split_f), "r") as lines:
                for line in lines:
                    line = line.rstrip()
                    words = line.split()
                    img_path = os.path.join(self.image_dir, words[0])

                    assert os.path.isfile(img_path), 'file %s error' % img_path
                    mask_path = img_path.replace(img_path.split('/')[-4], 'Mask')
                    assert os.path.isfile(mask_path), 'file %s error' % mask_path

                    if not dataset_cfg['real_training']:
                        if (int(words[1])==1):
                            self.images.append((img_path, mask_path, int(words[1])))
                    else:
                        self.images.append((img_path, mask_path, int(words[1])))
        elif self.datatype == 'test_vis':
            for osn in osn_list:
              _split_f = os.path.join(self.image_dir, 'test_'+ str(osn) +'_vis.txt')
              # obatin images
              with open(os.path.join(_split_f), "r") as lines:
                for line in lines:
                    line = line.rstrip()
                    words = line.split()
                    img_path = os.path.join(self.image_dir, words[0])

                    assert os.path.isfile(img_path), 'file %s error' % img_path
                    mask_path = img_path.replace(img_path.split('/')[-4], 'Mask')
                    assert os.path.isfile(mask_path), 'file %s error' % mask_path

                    if not dataset_cfg['real_training']:
                        if (int(words[1])==1):
                            self.images.append((img_path, mask_path, int(words[1])))
                    else:
                        self.images.append((img_path, mask_path, int(words[1])))
          

class FFDataset(BaseDataset):
    num_classes = 2
    classnames = ['real', 'fake']
    assert num_classes == len(classnames)
    clsid2label = {255: 1}
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(FFDataset, self).__init__(mode, logger_handle, dataset_cfg)
        # obtain the dirs
        rootdir = dataset_cfg['rootdir']
        data_name = dataset_cfg['type']
        self.compression = dataset_cfg['compression']
        fake_type = dataset_cfg['fake_type']
        self.image_dir = os.path.join(rootdir, data_name, self.compression)
        if self.mode.lower() == 'train':
            _split_f = os.path.join(rootdir, data_name, 'train.txt')
        elif self.mode.lower() == 'val':
            _split_f = os.path.join(rootdir, data_name, 'valid.txt')
        elif self.mode.lower() == 'test':
            _split_f = os.path.join(rootdir, data_name, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')
        # obatin images
        self.images = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                fake_name = line.rstrip()
                
                for src in fake_type:
                    filelist = glob.glob(os.path.join(self.image_dir, src, fake_name, '*.png'))
                    # if len(filelist) < 20:
                    #     print(f'{src},{fake_name},{len(filelist)}')

                    for fake_path in filelist:
                        assert os.path.isfile(fake_path)
                        mask_path = fake_path.replace(self.compression, 'Mask')
                        assert os.path.isfile(mask_path)
                        self.images.append((fake_path, mask_path, int(1)))
                        
                        if dataset_cfg['real_training']:
                            real_name = fake_name.split('_')[0]
                            real_id = os.path.basename(fake_path)
                            real_path = os.path.join(self.image_dir, 'youtube', real_name, real_id)
                            assert os.path.isfile(real_path)
                            mask_path = real_path.replace(self.compression, 'Mask')
                            assert os.path.isfile(mask_path)
                            self.images.append((real_path, mask_path, int(0)))