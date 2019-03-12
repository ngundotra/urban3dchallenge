import random

import numpy as np

from pytorch_utils.transforms import ToTensor
from dataset.abstract_image_provider import AbstractImageProvider
from .image_cropper import ImageCropper
from .reading_image_provider import ReadingImageProvider, MixedReadingImageProvider


class Dataset:
    """
    base class for pytorch datasets
    """
    def __init__(self, image_provider: AbstractImageProvider, image_indexes, config, stage='train', transforms=ToTensor()):
        self.cropper = ImageCropper(config.img_rows,
                                    config.img_cols,
                                    config.target_rows,
                                    config.target_cols,
                                    config.train_pad if stage=='train' else config.test_pad)
        self.image_provider = image_provider
        self.image_indexes = image_indexes if isinstance(image_indexes, list) else image_indexes.tolist()
        if stage != 'train' and len(self.image_indexes) % 2: #todo bugreport it
            self.image_indexes += [self.image_indexes[-1]]
        self.stage = stage
        self.keys = {'image', 'image_name'}
        self.config = config
        self.transforms = transforms
        if transforms is None:
            self.transforms = ToTensor()

    def __getitem__(self, item):
        raise NotImplementedError


class TrainDataset(Dataset):
    """
    dataset for train stage
    """
    def __init__(self, image_provider, image_indexes, config, stage='train', transforms=ToTensor()):
        super(TrainDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.keys.add('mask')

    def __getitem__(self, idx):
        im_idx = self.image_indexes[idx % len(self.image_indexes)]

        item = self.image_provider[im_idx]
        if type(self.image_provider) == ReadingImageProvider or self.image_provider.use_crop(im_idx):
            sx, sy = self.cropper.random_crop_coords()
            if self.cropper.use_crop and self.image_provider.has_alpha:
                for i in range(10):
                    alpha = self.cropper.crop_image(item.alpha, sx, sy)
                    if np.mean(alpha) > 5:
                        break
                    sx, sy = self.cropper.random_crop_coords()
                else:
                    return self.__getitem__(random.randint(0, len(self.image_indexes)))

            im = self.cropper.crop_image(item.image, sx, sy)
            mask = self.cropper.crop_image(item.mask, sx, sy)
        else:
            im = item.image
            mask = item.mask
        im, mask = self.transforms(im, mask)
        # cv2.imshow('w', np.moveaxis(im, 0, -1)[...,:3])
        # cv2.imshow('m', np.squeeze(mask))
        # cv2.waitKey()
        return {'image': im, 'mask': mask, 'image_name': item.fn}

    def __len__(self):
        return len(self.image_indexes) * max(self.config.epoch_size, 1) # epoch size is len images
    
    def is_mixed(self):
        return type(self.image_provider) == MixedReadingImageProvider

    def end_batch(self):
        """Cycles between the different datasets to provide from"""
        if self.is_mixed():
            splits = self.image_provider.get_splits()
            # s = np.random.choice(len(self.image_provider.ds_providers), p=np.array(splits)/len(self.image_provider)) 
            s = np.random.choice(len(self.image_provider.ds_providers))
            self.image_provider.set_split(s)
            print("New split is at:", self.image_provider.ds_idx)

class SequentialDataset(Dataset):
    """
    dataset for inference
    """
    def __init__(self, image_provider, image_indexes, config, stage='test', transforms=ToTensor()):
        super(SequentialDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.good_tiles = []
        self.init_good_tiles()
        self.keys.update({'sy', 'sx'})
        print(self.transforms)

    def init_good_tiles(self):
        self.good_tiles = []
        positions = self.cropper.positions
        for im_idx in self.image_indexes:
            if self.image_provider.has_alpha:
                item = self.image_provider[im_idx]
                alpha_generator = self.cropper.sequential_crops(item.alpha)
                for idx, alpha in enumerate(alpha_generator):
                    if np.mean(alpha) > 5:
                        self.good_tiles.append((im_idx, *positions[idx]))
            else:
                for pos in positions:
                    self.good_tiles.append((im_idx, *pos))

    def __getitem__(self, idx):
        if idx >= self.__len__():
            return None
        im_idx, sx, sy = self.good_tiles[idx]
        item = self.image_provider[im_idx]

        im = self.cropper.crop_image(item.image, sx, sy)

        im = self.transforms(im)
        # im = self.transforms(np.transpose(item.image, (2, 0, 1)))
        return {'image': im, 'startx': sx, 'starty': sy, 'image_name': item.fn}

    def __len__(self):
        return len(self.good_tiles)


class ValDataset(SequentialDataset):
    """
    dataset for validation
    """
    def __init__(self, image_provider, image_indexes, config, stage='train', transforms=ToTensor()):
        super(ValDataset, self).__init__(image_provider, image_indexes, config, stage, transforms=transforms)
        self.keys.add('mask')

    def __getitem__(self, idx):
        im_idx, sx, sy = self.good_tiles[idx]
        item = self.image_provider[im_idx]

        im = self.cropper.crop_image(item.image, sx, sy)
        mask = self.cropper.crop_image(item.mask, sx, sy)
        # cv2.imshow('w', im[...,:3])
        # cv2.imshow('m', mask)
        # cv2.waitKey()
        im, mask = self.transforms(im, mask)
        print(im.shape)
        return {'image': im, 'mask': mask, 'startx': sx, 'starty': sy, 'image_name': item.fn}
