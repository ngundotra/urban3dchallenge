import torch
import os

from dataset.reading_image_provider import ReadingImageProvider
from dataset.urban3d_dem_image import TiffDemImageType
from dataset.salmap_image import SalImageType
from dataset.tiff_image import TiffImageType
from dataset.crowdai_image import CrowdAIImageType
from dataset.spacenet_image import TiffSpacenetImageType 
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
parser.add_argument('dataset')
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

if args.dataset.lower() == 'spacenet':
    paths_testing = {
        'images': test_dir + '/RGB-PanSharpen/',
        'masks': test_dir + '/Shanghai-Segs/'
        }
    fn_mapping = {
            'masks': lambda name: 'buildings_AOI_4_Shanghai' + name.split('_')[-1] + '.geojson.npy', # FILL IN
        }
elif args.dataset.lower() == 'crowdai':
    paths_testing = {
            'images': os.path.join(test_data_path, 'images/'),
            'masks': os.path.join(test_data_path, '/labels/')
    }
    fn_mapping = {'masks': lambda name: name.replace("jpg", "npy")}


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

class TiffSpacenetImageTypeNoPad(TiffSpacenetImageType):
    def finalyze(self, data):
        return data

class CrowdAIImageTypeNoPad(CrowdAIImageType):
    def finalyze(self, data):
        return data

def predict():
    suffix = 'RGB'
    dtype = TiffDemImageTypeNopad
    if 'sal_map' in cfg and cfg['sal_map'] is True:
        dtype = TiffSalImageTypeNoPad
    if cfg['num_channels'] == 3:
        dtype = TiffImageTypeNoPad
    if args.dataset.lower() == 'spacenet':
        dtype = TiffSpacenetImageTypeNoPad
        suffix = 'tif'
    elif args.dataset.lower() == 'crowdai':
        dtype = CrowdAIImageTypeNoPad
        suffix = '.jpg'

    print(paths_testing, args.dataset.lower()=='crowdai')
    ds = ReadingImageProvider(dtype, paths_testing, fn_mapping, image_suffix=suffix)
    folds = [([], list(range(len(ds)))) for i in range(5)]

    num_workers = 0 if os.name == 'nt' else 4

    keval = GdalFullEvaluator(config, ds, folds, test=True, flips=3, num_workers=num_workers, border=0, crowdai=args.dataset.lower()=='crowdai')
    keval.predict()


if __name__ == "__main__":
    if args.dataset.lower() == 'spacenet':
        print("Spacenet is true")
        print(os.path.join(config.results_dir, 'results', config.folder, 'merged'), test_data_path, out_file)

    print(config.num_channels)
    in_shape = 2048
    test_shape = 2048
    if args.dataset.lower() == 'spacenet':
        in_shape = 650
        test_shape = 512
    elif args.dataset.lower() == 'crowdai':
        in_shape = 300
        test_shape = 300
    config = update_config(config, img_rows=in_shape, img_cols=in_shape, target_rows=test_shape, target_cols=test_shape, num_channels=config.num_channels)
    print("predicting stage 1/3")
    predict()

    print("predicting stage 2/3")
    merge_tiffs(os.path.join(config.results_dir, 'results', config.folder))
    print("predicting stage 3/3")
    make_submission(os.path.join(config.results_dir, 'results', config.folder, 'merged'), test_data_path, out_file, spacenet=args.dataset.lower())
