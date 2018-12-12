import os

import numpy as np
from osgeo import gdal

from .abstract_image_type import AbstractImageType


class TiffImageType(AbstractImageType):
    "image type for reading images with gdal"
    def __init__(self, paths, fn, fn_mapping, has_alpha):
        # fn stands for filename
        super().__init__(paths, fn, fn_mapping, has_alpha)
        self.src_ds = None
        self.mask_ds = None

    def read_image(self):
        if self.src_ds is None:
            self.src_ds = gdal.Open(os.path.join(self.paths['images'], self.fn), gdal.GA_ReadOnly)
        # I have no idea what this does
        last_channel = self.src_ds.RasterCount + (1 if not self.has_alpha else 0)
        # I'm assuming this gets the channels of the image...?
        arr = [self.src_ds.GetRasterBand(idx).ReadAsArray() for idx in range(1, last_channel)]
        return self.finalyze(np.dstack(arr))

    def read_mask(self):
        if self.mask_ds is None:
            # This parses the mask file name from the RGB filename
            self.mask_ds = gdal.Open(os.path.join(self.paths['masks'], self.fn_mapping['masks'](self.fn)), gdal.GA_ReadOnly)
        mask = self.mask_ds.GetRasterBand(1).ReadAsArray()
        mask = (mask > 0).astype(np.uint8) * 255
        return self.finalyze(mask)

    def read_alpha(self):
        if self.src_ds is None:
            # This just reads in the filename & reads the alpha channel, I think
            self.src_ds = gdal.Open(os.path.join(self.paths['images'], self.fn), gdal.GA_ReadOnly)
        return self.finalyze(self.src_ds.GetRasterBand(self.src_ds.RasterCount).ReadAsArray())

    def finalyze(self, data):
        return self.reflect_border(data)
