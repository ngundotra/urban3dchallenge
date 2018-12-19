import os

from src.dataset.reading_image_provider import ReadingImageProvider
from src.dataset.urban3d_dem_image import TiffDemImageType
from src.dataset.tiff_image import TiffImageType

TRAIN_PATH='../Train/Inputs/'
dataset_path, train_dir = os.path.split(TRAIN_PATH)

paths = {
    'masks': train_dir,
    'images': train_dir,
    'dems': train_dir,
    'dtms': train_dir,
}

fn_mapping = {
    'masks': lambda name: name.replace('RGB', 'GTI'), # okay so truths are GTI
    'dems': lambda name: name.replace('RGB', 'DSM'), # dems = DSMs
    'dtms': lambda name: name.replace('RGB', 'DTM') # dtms = DTMs
}

paths = {k: os.path.join(dataset_path, v) for k,v in paths.items()}

ds = ReadingImageProvider(TiffImageType, paths, fn_mapping,
            image_suffix='RGB'
)
