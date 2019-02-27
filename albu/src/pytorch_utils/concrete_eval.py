import os

import cv2
import numpy as np
from osgeo import gdal
from osgeo.gdalnumeric import CopyDatasetInfo

from .eval import Evaluator
import shutil

class FullImageEvaluator(Evaluator):
    """
    Mixin for processing not crops
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_data(self, predicted, model, data, prefix=""):
        names, samples, masks = self.get_data(data)
        for i in range(len(names)):
            self.prev_name = names[i]
            self.full_pred = np.squeeze(predicted[i,...])
            if samples is not None:
                self.full_image = (samples[i,...] * 255).astype(np.uint8)
            if masks is not None:
                self.full_mask = (np.squeeze(masks[i,...]) * 255).astype(np.uint8)
            self.on_image_constructed(prefix)

    def save(self, name, prefix=""):
        cv2.imwrite(os.path.join(self.config.results_dir, 'results', self.config.folder, 'mask_{}'.format(name)), (self.full_pred * 255).astype(np.uint8))


class GdalSaver(Evaluator):
    """
    Mixin for gdal data type
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.paths = self.ds.paths
        path = os.path.join(self.config.results_dir, 'results', self.config.folder)
        shutil.rmtree(path, True)
        self.crowdai_save_dir='/datasets/CrowdAI/predictions/'
        if self.crowdai:
            self.crowdai_save_dir='/datasets/CrowdAI/predictions/'
            self.save_dir = os.path.join(self.crowdai_save_dir, self.config.results_dir)
            path = os.path.join(self.save_dir, 'results', self.config.folder)
        os.makedirs(path, exist_ok=True)

    def save(self, name, prefix=""):
        #todo has mask
        if not self.crowdai:
            has_mask = False
            ref_file = os.path.join(self.paths['images'], name)
            res_path_geo = os.path.join(self.config.results_dir, 'results', self.config.folder, prefix + name)
            driver = gdal.GetDriverByName('GTiff')
            outRaster = driver.Create(res_path_geo, self.full_pred.shape[1], self.full_pred.shape[0], 1, gdal.GDT_Float32)
            if os.path.isfile(ref_file):
                gdalData = gdal.Open(ref_file, gdal.GA_ReadOnly)
                nodata = gdalData.GetRasterBand(1).GetNoDataValue()
                if has_mask:
                    mask = np.array(gdalData.GetRasterBand(4).ReadAsArray())
                    empty_mask = np.zeros((self.full_pred.shape[0], self.full_pred.shape[1]))
                    empty_mask[0:mask.shape[0], 0:mask.shape[1]] = mask
                    empty_mask = np.invert(empty_mask.astype(np.bool))
                    self.full_pred[empty_mask] = nodata
                geoTrans = gdalData.GetGeoTransform()
                outRaster.SetGeoTransform(geoTrans)
                CopyDatasetInfo(gdalData, outRaster)
            outband = outRaster.GetRasterBand(1)
            outband.WriteArray(self.full_pred)
            outRaster.FlushCache()
        else:
            res_path_geo = os.path.join(self.save_dir, 'results', self.config.folder, prefix + name.replace('jpg', 'npy'))
            np.save(res_path_geo, self.full_pred)


class GdalFullEvaluator(GdalSaver, FullImageEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save(self, name, prefix=""):
        GdalSaver.save(self, name, prefix)
