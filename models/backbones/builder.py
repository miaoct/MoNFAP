'''
Function:
    Implementation of BackboneBuilder and BuildBackbone
Author:
    Zhenchao Jin
'''
import copy
from utils import BaseModuleBuilder
from .convnextv2 import ConvNeXtV2


'''BackboneBuilder'''
class BackboneBuilder(BaseModuleBuilder):
    REGISTERED_MODULES = {
        'ConvNeXtV2': ConvNeXtV2,
    }
    '''build'''
    def build(self, backbone_cfg):
        backbone_cfg = copy.deepcopy(backbone_cfg)
        if 'selected_indices' in backbone_cfg: backbone_cfg.pop('selected_indices')
        return super().build(backbone_cfg)


'''BuildBackbone'''
BuildBackbone = BackboneBuilder().build