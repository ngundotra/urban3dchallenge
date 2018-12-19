import torch
import os

from pytorch_utils.transforms import augment_flips_color

from dataset.reading_image_provider import ReadingImageProvider
from dataset.urban3d_dem_image import TiffDemImageType
from dataset.tiff_image import TiffImageType
from pytorch_utils.train import train
from utils import get_folds, update_config
import argparse
import json
from config import Config

parser = argparse.ArgumentParser()
# This is the path to the JSON that he uses to parse the options
# for training a net
parser.add_argument('config_path')
# This is assumed to be "data/training" from the previous JSON, this overwrites the JSON ^ above
# So if its not specified, then the config path above will specify the data path
parser.add_argument('train_data_path')
args = parser.parse_args()
with open(args.config_path, 'r') as f:
    cfg = json.load(f)
    dataset_path, train_dir = os.path.split(cfg['dataset_path'])
    cfg['dataset_path'] = dataset_path
config = Config(**cfg)

masks_dir = os.path.join(train_dir, 'GT')
rgb_dir = os.path.join(train_dir, 'Inputs')
paths = {
    'masks': masks_dir,
    'images': rgb_dir,
    'dems': rgb_dir,
    'dtms': rgb_dir,
}

fn_mapping = {
    'masks': lambda name: name.replace('RGB', 'GTI'), # okay so truths are GTI
    'dems': lambda name: name.replace('RGB', 'DSM'), # dems = DSMs
    'dtms': lambda name: name.replace('RGB', 'DTM') # dtms = DTMs
}

paths = {k: os.path.join(config.dataset_path, v) for k,v in paths.items()}


def train_stage0():
    """
    heat up weights for 5 epochs
    """
    ds = ReadingImageProvider(TiffImageType, paths, fn_mapping, image_suffix='RGB')

    folds = get_folds(ds, 5)
    num_workers = 0 if os.name == 'nt' else 8
    train(ds, folds, config, num_workers=num_workers, transforms=augment_flips_color)


def train_stage1():
    """
    main training stage with dtm/dsm data
    """
    ds = ReadingImageProvider(TiffDemImageType, paths, fn_mapping, image_suffix='RGB')

    folds = get_folds(ds, 5)
    num_workers = 0 if os.name == 'nt' else 8
    train(ds, folds, config, num_workers=num_workers, transforms=augment_flips_color, num_channels_changed=True)


def train_stage2():
    """
    train with other loss function
    """
    ds = ReadingImageProvider(TiffDemImageType, paths, fn_mapping, image_suffix='RGB')

    folds = get_folds(ds, 5)
    num_workers = 0 if os.name == 'nt' else 8
    train(ds, folds, config, num_workers=num_workers, transforms=augment_flips_color)


if __name__ == "__main__":
    num_epochs = config.nb_epoch
    config = update_config(config, num_channels=3, nb_epoch=5)
    print("start training stage 1/3")
    train_stage0()
    print("start training stage 2/3")
    config = update_config(config, num_channels=4, nb_epoch=num_epochs)
    train_stage1()
    print("start training stage 3/3")
    config = update_config(config, loss=config.loss + '_w', nb_epoch=num_epochs + 2)
    train_stage2()
