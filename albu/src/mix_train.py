import torch
import os

from pytorch_utils.transforms import augment_flips_color

from dataset.reading_image_provider import ReadingImageProvider, MixedReadingImageProvider
from dataset.urban3d_dem_image import TiffDemImageType
from dataset.salmap_image import SalImageType
from dataset.tiff_image import TiffImageType
from dataset.spacenet_image import TiffSpacenetImageType
from dataset.crowdai_image import CrowdAIImageType
from pytorch_utils.train import mix_train
from utils import get_folds, update_config
import argparse
import json
from config import Config


def get_training_mappings(data_info, sal_map=False):
    """Returns the paths & fn_mappings for a given dataset & data_dir"""
    ds_name, data_dir = data_info['name'], data_info['dataset_path']
    limit = data_info['limit']

    if ds_name == 'urban3d':
        image_type = TiffImageType
        if sal_map:
            image_type = SalImageType
        masks_dir = os.path.join(data_dir, 'GT') 
        rgb_dir = os.path.join(data_dir, 'Inputs')
        paths_training = {
            'masks': masks_dir,
            'images': rgb_dir,
            'dems': rgb_dir,
            'dtms': rgb_dir,
            'sal': rgb_dir,
        }
        fn_mapping = {
            'masks': lambda name: name.replace('RGB', 'GTI'),
            'dems': lambda name: name.replace('RGB', 'DSM'),
            'dtms': lambda name: name.replace('RGB', 'DTM'),
            'sal': lambda name: name[:-7] + 'SAL.npy'
        }
        image_suffix = 'RGB'
    elif ds_name == 'spacenet':
        image_type = TiffSpacenetImageType
        paths_training = {
            'images': os.path.join(data_dir, 'RGB-PanSharpen/'),
            'masks': os.path.join(data_dir, 'Shanghai-Segs/')
            }
        fn_mapping = {
                'masks': lambda name: 'buildings_AOI_4_Shanghai_' + name.split('_')[-1].replace("tif", 'geojson.npy'), # FILL IN
            }
        image_suffix = 'tif'
    elif ds_name == 'crowdai':
        image_type = CrowdAIImageType
        paths_training = {
                'images': os.path.join(data_dir, 'images/'),
                'masks': os.path.join(data_dir, 'labels/')
        }
        image_suffix = 'jpg'
        fn_mapping = {'masks': lambda name: name.replace("jpg", "npy")}
    else:
        raise ValueError("Dataset Name not recognized: {}".format(ds_name))
    paths = paths_training
    rip_info = {'paths': paths, 'fn_mapping': fn_mapping, 'image_type': image_type,
                 'image_suffix': image_suffix, "limit": limit}
    return rip_info
def train_stage0():
    """
    heat up weights for 5 epochs
    """
    ds = make_mixed_ds()

    num_workers = 0 if os.name == 'nt' else WORKERS
    tfs = [None if type(prov)==ReadingImageProvider else augment_flips_color for prov in ds.ds_providers]
    mix_train(ds.ds_providers, config, num_workers=num_workers, transforms=tfs)


def train_stage1(sal_map:bool, three=False):
    """
    main training stage with dtm/dsm data
    three = True ===> use only RGB for training
    updates channels from warm start with only RGB to final number of channels in config.num_channels
    """
    ds = make_mixed_ds()

    tfs = [None if type(prov)==ReadingImageProvider else augment_flips_color for prov in ds.ds_providers]
    num_workers = 0 if os.name == 'nt' else WORKERS
    mix_train(ds.ds_providers, config, num_workers=num_workers, transforms=tfs, num_channels_changed=not three)


def train_stage2(sal_map:bool, three=False):
    """
    train with other loss function
    three = True ===> use only RGB for training
    """
    ds = make_mixed_ds()

    tfs = [None if type(prov)==ReadingImageProvider else augment_flips_color for prov in ds.ds_providers]
    num_workers = 0 if os.name == 'nt' else WORKERS
    mix_train(ds.ds_providers, config, num_workers=num_workers, transforms=tfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # This is the path to the JSON that he uses to parse the options
    # for training a net
    parser.add_argument('config_path')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        cfg = json.load(f)
    if len(cfg['datasets']) == 1:
        raise ValueError("USE REGULAR TRAIN WHEN TRAINING ONLY ON 1 DATASET")
    elif len(cfg['datasets']) >= 4:
        raise ValueError("TOO MANY DATASETS PASSED IN, ONLY SUPPORT <= 3")
    datasets = cfg.pop('datasets')

    sal_map = cfg.get("sal_map", False)
    data_infos = [get_training_mappings(data, sal_map) for data in datasets]
    make_mixed_ds = lambda: MixedReadingImageProvider(data_infos)

    FOLDS = 5
    WORKERS = 2
    # Note Sal_map usage is a property that should affect all datasets,
    # since it implies that the model is using 4 channels
    config = Config(**cfg)
    three = (config.num_channels == 3)
    num_epochs = config.nb_epoch
    config = update_config(config, num_channels=3, nb_epoch=5)
    print("start training stage 1/3")
    train_stage0()
    print("start training stage 2/3")
    config = update_config(config, num_channels=config.num_channels, nb_epoch=num_epochs)
    train_stage1(config.sal_map, three=three)
    print("start training stage 3/3")
    config = update_config(config, loss=config.loss + '_w', nb_epoch=num_epochs + 2)
    train_stage2(config.sal_map, three=three)
