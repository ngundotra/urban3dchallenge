import os 
 
import numpy as np
from osgeo import gdal

from .abstract_image_type import AbstractImageType


class SalImageType(AbstractImageType):
    """
    Manages files to RGB, Saliency maps, and the ground truth files (called masks)
    Used to retrain albu's solution on saliency maps. 
    
    Requires 
        `fn_mapping['sal'] = lambda x: x[:-7] + 'SAL.png'`
    """
    def __init__(self, paths, fn, fn_mapping, has_alpha):
        super().__init__(paths, fn, fn_mapping, has_alpha)
        self.src_ds = None
        self.mask_ds = None
        self.sal_ds = None

    def read_image(self):
        if self.src_ds is None:
            self.src_ds = gdal.Open(os.path.join(self.paths['images'], self.fn), gdal.GA_ReadOnly)
        if self.sal_ds is None:
            self.sal_ds = np.load(os.path.join(self.paths['sal'], self.fn_mapping['sal'](self.fn)))
        last_channel = self.src_ds.RasterCount + (1 if not self.has_alpha else 0)
        arr = [self.src_ds.GetRasterBand(idx).ReadAsArray() for idx in range(1, last_channel)]

        arr = [(a / 255.).astype(np.float32) for a in arr]
        arr.append(self.sal_ds)
        return self.finalyze(np.dstack(arr))

    def read_mask(self):
        if self.mask_ds is None:
            self.mask_ds = gdal.Open(os.path.join(self.paths['masks'], self.fn_mapping['masks'](self.fn)), gdal.GA_ReadOnly)
        mask = self.mask_ds.GetRasterBand(1).ReadAsArray()
        mask = (mask > 0).astype(np.uint8) * 255
        return self.finalyze(mask)

    def read_alpha(self):
        if self.src_ds is None:
            self.src_ds = gdal.Open(os.path.join(self.paths['images'], self.fn), gdal.GA_ReadOnly)
        return self.finalyze(self.src_ds.GetRasterBand(self.src_ds.RasterCount).ReadAsArray())

    def finalyze(self, data):
        return self.reflect_border(data)

