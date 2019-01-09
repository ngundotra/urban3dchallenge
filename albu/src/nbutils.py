import cv2
import numpy as np
import gdal
from scipy.spatial.distance import dice

def load_img(src, tiff=False):
    has_alpha = 0
    src_ds = gdal.Open(src, gdal.GA_ReadOnly)
    if tiff:
        return src_ds
    last_channel = src_ds.RasterCount + (1 if not has_alpha else 0)
    arr = [src_ds.GetRasterBand(idx).ReadAsArray() for idx in range(1, last_channel)]
    return np.dstack(arr)

def thresh(x):
    """Use this to refine predictions on the (0, 255) range of type np.uint8"""
    retval, dst = cv2.threshold(x, 127, 255, 0)
    return dst

def score_pred(pred, truth):
    return 1 - dice(thresh(pred).flatten() > 1, truth.flatten() >= 1)

def make_input_tensor(img, dem):
    def big(x, axis=-1):
        return np.expand_dims(x, axis=axis)
                
    input_tensor = np.dstack([big(img[:,:,0]), big(img[:,:,1]), 
                              big(img[:,:,2]), big(dem)])
    return np.expand_dims(np.transpose(input_tensor, (2, 0, 1)), axis=0)
