import os
import math

import numpy as np
import cv2
from osgeo import gdal

from .abstract_image_type import AbstractImageType

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix

def simplest_cb(img, percent=1):
    """Colorbalances images taken from: https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc"""

    assert img.shape[2] == 3
    assert percent > 0 and percent < 100
    half_percent = percent / 200.0
    channels = cv2.split(img)

    out_channels = []
    for channel in channels:
        assert len(channel.shape) == 2
        # find the low and high precentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)
        assert len(flat.shape) == 1
        flat = np.sort(flat)

        n_cols = flat.shape[0]
        low_val  = flat[int(math.floor(n_cols * half_percent))]                                                                                                                                             
        high_val = flat[int(math.ceil( n_cols * (1.0 - half_percent)))]
        # saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)
        # scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)                
        out_channels.append(normalized)
    return cv2.merge(out_channels)


class TiffSpacenetImageType(AbstractImageType):
    """
    image type, that works with SpaceNet images
    """
    def __init__(self, paths, fn, fn_mapping, has_alpha):
        super().__init__(paths, fn, fn_mapping, has_alpha)
        self.src_ds = None
        self.mask_ds = None

    def read_image(self):
        """Apply percent-based image preprocessing to shape range from [0, 2^11] to [0, 1]"""
        if self.src_ds is None:
            self.src_ds = gdal.Open(os.path.join(self.paths['images'], self.fn), gdal.GA_ReadOnly)
        last_channel = self.src_ds.RasterCount + (1 if not self.has_alpha else 0)
        arr = [self.src_ds.GetRasterBand(idx).ReadAsArray() for idx in range(1, last_channel)]

        arr = np.dstack([(a).astype(np.float32) for a in arr])
        arr = simplest_cb(arr).astype(np.uint8)
        # Max is 255, min is 0
        return self.finalyze(arr)

    def read_mask(self):
        if self.mask_ds is None:
            self.mask_ds = np.load(os.path.join(self.paths['masks'], self.fn_mapping['masks'](self.fn)))
        self.mask_ds = (self.mask_ds > 0).astype(np.uint8) * 255
        return self.finalyze(self.mask_ds)

    def finalyze(self, data):
        return self.reflect_border(data)

