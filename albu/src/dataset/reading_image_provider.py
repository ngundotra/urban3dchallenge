import os

from .abstract_image_provider import AbstractImageProvider


class ReadingImageProvider(AbstractImageProvider):
    """
    provides images for dataset from disk
    """
    def __init__(self, image_type, paths, fn_mapping, image_suffix=None, has_alpha=False):
        super(ReadingImageProvider, self).__init__(image_type, fn_mapping, has_alpha=has_alpha)
        # The super init stores the fn mapping, also makes use of `typing` module in python
        # to enforce the types of arguments passed into it. Note he only defines 1 image type lmao
        self.im_names = os.listdir(paths['images'])
        # Usually grabs only RGB images to store into im_names
        if image_suffix is not None:
            # Honestly it should be `if n.endswith(image_suffix)` in case RGB somewhere in there
            self.im_names = [n for n in self.im_names if image_suffix in n]

        self.paths = paths

    def __getitem__(self, item):
        # Oh whack, the image type creates the filename from the im_names & fn_mapping
        return self.image_type(self.paths, self.im_names[item], self.fn_mapping, self.has_alpha)

    def __len__(self):
        # This is the number of inputs-
        # ah i see, this is because he's recreating the whole input system from the RGB file
        # and then linking the DSM/DTM/GT files with the same name, classico
        return len(self.im_names)