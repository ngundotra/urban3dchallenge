import os

import numpy as np
from osgeo import gdal

from .abstract_image_type import AbstractImageType


class TiffDemImageType(AbstractImageType):
    """
    image type, that has dem/dtm information
    """
    def __init__(self, paths, fn, fn_mapping, has_alpha):
        super().__init__(paths, fn, fn_mapping, has_alpha)
        self.src_ds = None
        self.mask_ds = None
        self.dem_ds = None
        self.dtm_ds = None

    def read_image(self):
        if self.src_ds is None:
            self.src_ds = gdal.Open(os.path.join(self.paths['images'], self.fn), gdal.GA_ReadOnly)
        if self.dem_ds is None:
            self.dem_ds = gdal.Open(os.path.join(self.paths['dems'], self.fn_mapping['dems'](self.fn)), gdal.GA_ReadOnly)
        if self.dtm_ds is None:
            self.dtm_ds = gdal.Open(os.path.join(self.paths['dtms'], self.fn_mapping['dtms'](self.fn)), gdal.GA_ReadOnly)
        last_channel = self.src_ds.RasterCount + (1 if not self.has_alpha else 0)
        arr = [self.src_ds.GetRasterBand(idx).ReadAsArray() for idx in range(1, last_channel)]

        band_dem = self.dem_ds.GetRasterBand(1)
        band_dtm = self.dtm_ds.GetRasterBand(1)
        nodata = band_dem.GetNoDataValue()
        dem = band_dem.ReadAsArray()
        dtm = band_dtm.ReadAsArray()
        dem[dem==nodata] = 0
        dtm[dtm==nodata] = 0
        dem -= dtm
        dem /= 9.

        arr = [(a / 255.).astype(np.float32) for a in arr]
        arr.append(dem)
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

