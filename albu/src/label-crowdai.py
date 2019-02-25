import numpy as np
import sys, os 
import skimage.io as io
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as cocomask


crowddir = '/datasets/CrowdAI/Train/'
datadir = crowddir+'images/'
lbl_big = crowddir+'annotation.json'

coco = COCO(lbl_big)

category_ids = coco.loadCats(coco.getCatIds())
image_ids = coco.getImgIds(catIds=coco.getCatIds())

for img_id in tqdm(image_ids):
    img = coco.loadImgs([img_id])[0]
    image_path = os.path.join(datadir, img["file_name"])
    I = io.imread(image_path)
    annotation_ids = coco.getAnnIds(imgIds=img['id'])
    annotations = coco.loadAnns(annotation_ids)
    rle = cocomask.frPyObjects(annotations[0]['segmentation'], img['height'], img['width'])
    m = cocomask.decode(rle)
    # m.shape has a shape of (300, 300, 1)
    # so we first convert it to a shape of (300, 300)
    m = m.reshape((img['height'], img['width']))
    np.save(crowddir+'labels/'+"{:0=12}.npy".format(img_id), m)
