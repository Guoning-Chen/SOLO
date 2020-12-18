import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS
from mmdet.datasets.coco import CocoDataset


@DATASETS.register_module
class WearInstanceDataset(CocoDataset):

    CLASSES = ('chain', 'oil', 'fiber')