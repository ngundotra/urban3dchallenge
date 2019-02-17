import torch
import os

from dataset.reading_image_provider import ReadingImageProvider
from dataset.urban3d_dem_image import TiffDemImageType
from dataset.salmap_image import SalImageType
from dataset.tiff_image import TiffImageType
from pytorch_utils.concrete_eval import GdalFullEvaluator
from utils import update_config
# torch.backends.cudnn.benchmark = True
import argparse
import json
from config import Config
from other_tools.merge_preds import merge_tiffs
from other_tools.make_submission import make_submission


parser = argparse.ArgumentParser()
parser.add_argument('config_path')
parser.add_argument('train_data_path')
parser.add_argument('test_data_path')
parser.add_argument('output_file')
args = parser.parse_args()
with open(args.config_path, 'r') as f:
    cfg = json.load(f)
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    out_file = args.output_file
    dataset_path, test_dir = os.path.split(test_data_path)
    cfg['dataset_path'] = dataset_path
config = Config(**cfg)

paths_testing = {
    'masks': test_dir,
    'images': test_dir,
    'dems': test_dir,
    'dtms': test_dir,
    'sal': test_dir
}
fn_mapping = {
    'masks': lambda name: name.replace('RGB', 'GTI'),
    'dems': lambda name: name.replace('RGB', 'DSM'),
    'dtms': lambda name: name.replace('RGB', 'DTM'),
    'sal': lambda name: name[:-7] + 'SAL.npy'
}

paths_testing = {k: os.path.join(config.dataset_path, v) for k,v in paths_testing.items()}

class TiffImageTypeNoPad(TiffImageType):
    def finalyze(self, data):
        return data


class TiffSalImageTypeNoPad(SalImageType):
    def finalyze(self, data):
        return data


class TiffDemImageTypeNopad(TiffDemImageType):
    def finalyze(self, data):
        return data


def predict():
    dtype = TiffDemImageTypeNopad
    if 'sal_map' in cfg and cfg['sal_map'] is True:
        dtype = TiffSalImageTypeNoPad
    if cfg['num_channels'] == 3:
        dtype = TiffImageTypeNoPad
    ds = ReadingImageProvider(dtype, paths_testing, fn_mapping, image_suffix='RGB')
    folds = [([], list(range(len(ds)))) for i in range(5)]

    num_workers = 0 if os.name == 'nt' else 4

    keval = GdalFullEvaluator(config, ds, folds, test=True, flips=3, num_workers=num_workers, border=0)
    keval.predict()


if __name__ == "__main__":
    config = update_config(config, img_rows=2048, img_cols=2048, target_rows=2048, target_cols=2048, num_channels=4)
    print("predicting stage 1/3")
    predict()

    print("predicting stage 2/3")
    merge_tiffs(os.path.join(config.results_dir, 'results', config.folder))
    print("predicting stage 3/3")
    make_submission(os.path.join(config.results_dir, 'results', config.folder, 'merged'), test_data_path, out_file)
