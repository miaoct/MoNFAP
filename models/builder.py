'''
Function:
    Implementation of SegmentorBuilder and BuildSegmentor
Author:
    Changtao Miao
'''
import copy
from utils import BaseModuleBuilder
from .forgery2mask import Forgery2Mask

'''SegmentorBuilder'''
class SegmentorBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'Forgery2Mask': Forgery2Mask,
    }
    '''build'''
    def build(self, segmentor_cfg, mode):
        segmentor_cfg = copy.deepcopy(segmentor_cfg)
        segmentor_type = segmentor_cfg.pop('type')
        segmentor = self.REGISTERED_MODULES[segmentor_type](cfg=segmentor_cfg, mode=mode)
        return segmentor


'''BuildSegmentor'''
BuildSegmentor = SegmentorBuilder().build