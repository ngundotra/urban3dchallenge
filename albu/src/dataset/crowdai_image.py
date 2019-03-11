import os 
 
import numpy as np
import skimage.io as io
import cv2

from .abstract_image_type import AbstractImageType


class CrowdAIImageType(AbstractImageType):
    """
    Resizes CrowdAI data to be (256, 256) from original (300, 300) to work
    with the model architecture
    """
    def __init__(self, paths, fn, fn_mapping, has_alpha):
        super().__init__(paths, fn, fn_mapping, has_alpha)
        self.fn_mapping = lambda x: x.replace('jpg', 'npy')
        self.src_ds = None
        self.mask_ds = None

    def read_image(self):
        if self.src_ds is None:
            self.src_ds = io.imread(os.path.join(self.paths['images'], self.fn))
            self.src_ds = cv2.resize(self.src_ds, (256, 256))
        return self.finalyze(self.src_ds)

    def read_mask(self):
        if self.mask_ds is None:
            self.mask_ds = np.load(os.path.join(self.paths['masks'], self.fn_mapping['masks'](self.fn)))
            self.mask_ds = cv2.resize(self.mask_ds, (256, 256))
        mask = (self.mask_ds > 0).astype(np.uint8) * 255
        return self.finalyze(mask)

    def finalyze(self, data):
        return self.reflect_border(data)

