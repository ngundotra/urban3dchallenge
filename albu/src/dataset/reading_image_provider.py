import os

from .abstract_image_provider import AbstractImageProvider
from .tiff_image import TiffImageType


class ReadingImageProvider(AbstractImageProvider):
    """
    provides images for dataset from disk
    """
    def __init__(self, image_type, paths, fn_mapping, image_suffix=None, has_alpha=False, limit=None):
        super(ReadingImageProvider, self).__init__(image_type, fn_mapping, has_alpha=has_alpha)
        # The super init stores the fn mapping, also makes use of `typing` module in python
        # to enforce the types of arguments passed into it. Note he only defines 1 image type lmao
        self.im_names = os.listdir(paths['images'])

        if image_suffix is not None:
            # Honestly it should be `if n.endswith(image_suffix)` in case RGB somewhere in there
            self.im_names = [n for n in self.im_names if image_suffix in n]

        # Usually grabs only RGB images to store into im_names
        if limit and type(limit) == int:
            self.im_names = self.im_names[:limit]
        elif limit and type(limit) == float:
            self.im_names = self.im_names[:int(limit*len(self.im_names))]

        print(limit, len(self))

        self.paths = paths

    def __getitem__(self, item):
        # Oh whack, the image type creates the filename from the im_names & fn_mapping
        return self.image_type(self.paths, self.im_names[item], self.fn_mapping, self.has_alpha)

    def __len__(self):
        # This is the number of inputs-
        # ah i see, this is because he's recreating the whole input system from the RGB file
        # and then linking the DSM/DTM/GT files with the same name, classico
        return len(self.im_names)
    
    def is_mixed(self):
        return False


class MixedReadingImageProvider():
    """Class explicity for providing a mixed batch of images, shuffled randomly"""
    def __init__(self, datasets):
        """The datasets argument should be a dictionary of arguments that we can directly create
        a ReadingImageProvider from:
        datasets = [
            {'image_type': Class here,
            'paths': ...,
            'fn_mapping': ...,
            'image_suffix': ...,
            'has_alpha': ...,
            'limit': int/float},
            ...
        ]"""
        self.has_alpha = False
        self.ds_providers = []
        for data_info in datasets:
            self.ds_providers.append(ReadingImageProvider(**data_info))
        print(len(self), self.get_splits())

    def __getitem__(self, item):
        # OOF, the asynchronous loading messes up logic quite a bit
        print(item)
        total = 0
        for ds in self.ds_providers:
            if item < total + len(ds):
                return ds.image_type(ds.paths, ds.im_names[item - total], ds.fn_mapping, ds.has_alpha)
            total += len(ds)

    def set_batch_size(self, bs):
        for ds in self.ds_providers:
            ds.im_names[:len(ds.im_names)//bs*bs]

    def use_crop(self, item):
        """Returns True if Urban3D, otherwise False"""
        total = 0
        for ds in self.ds_providers:
            total += len(ds)
            if item < total and ds.image_type == TiffImageType:
                return True
        return False
    
    def is_mixed(self):
        return True

    def __len__(self):
        return sum([len(ds) for ds in self.ds_providers])

    def get_splits(self):
        """Returns the lengths of each dataset"""
        total = 0
        splits = []
        for ds in self.ds_providers:
            total += len(ds)
            splits.append(total)
        return splits
    
    def set_split(self, ds_idx):
        """Sets all get_items to be modular within a specific dataset"""
        if ds_idx == -1:
            self.ds_idx = -1
        elif ds_idx < len(self.ds_providers):
            self.ds_idx = ds_idx
        else:
            raise ValueError("set_split: invalid dataset index passed {} is too large".format(ds_idx))