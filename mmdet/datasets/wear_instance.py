from .coco import CocoDataset
from .registry import DATASETS


@DATASETS.register_module
class WearInstanceDataset(CocoDataset):

    CLASSES = ('chain', 'block', 'cutting', 'sphere', 'oxide')