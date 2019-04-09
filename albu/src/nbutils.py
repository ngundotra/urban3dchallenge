"""nbutils.py
This file was created to store some of the more gross functions I had to write when doing early Notebooking
for this project. If you don't like how it's structured, you can email complaints to /dev/null. Hahaha, actually
I'm sorry if you do have complaints."""

import torch
import cv2
import georaster
import numpy as np
import gdal
from matplotlib.collections import PatchCollection
from matplotlib.pyplot import Polygon
from pytorch_utils.eval import flip, predict
from pytorch_utils.loss import dice
from scipy.spatial.distance import dice as scipy_dice

def load_img(src, tiff=False):
    has_alpha = 0
    src_ds = gdal.Open(src, gdal.GA_ReadOnly)
    if tiff:
        return src_ds
    last_channel = src_ds.RasterCount + (1 if not has_alpha else 0)
    arr = [src_ds.GetRasterBand(idx).ReadAsArray() for idx in range(1, last_channel)]
    return np.dstack(arr)

def thresh(x):
    """Use this to refine predictions on the (0, 255) range of type np.uint8"""
    retval, dst = cv2.threshold(x, 127, 255, 0)
    return dst

def score_pred(pred, truth):
    return 1 - scipy_dice(thresh(pred).flatten() > 1, truth.flatten() >= 1)

def make_input_tensor(img, dem):
    def big(x, axis=-1):
        return np.expand_dims(x, axis=axis)

    input_tensor = np.dstack([big(img[:,:,0]), big(img[:,:,1]),
                              big(img[:,:,2]), big(dem)])
    return np.expand_dims(np.transpose(input_tensor, (2, 0, 1)), axis=0)

def get_saliency(model, input_tensor, truth, verbose=False):
    """Backpropagates loss to input and returns absolute value of the input gradient
    Meant for input_tensors of size <= [4, 1024, 1024]"""
    input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)
    output = torch.sigmoid(model(input_tensor.cuda()))
    loss = 1 - dice(output, torch.Tensor(truth).cuda())
    loss.backward()

    saliency = np.transpose(input_tensor.grad.data.cpu().numpy()[0], (1, 2, 0))
    abs_sal = np.abs(saliency).max(axis=2)
#     abs_sal = np.abs(saliency)[...,3]
    if verbose:
        if truth is None:
            raise ValueError("Need to pass in truth labels in range [0, 1].")
        pred = output.detach().cpu().numpy().squeeze()
        scipy_dice = score_pred((pred*255).astype(np.uint8), truth)
        print("Torch loss:", loss, '\nDice:', scipy_dice)
    return abs_sal

def get_saliency_big(model, input_tensor, truth, verbose=False):
    """Takes in input tensor of size [4, 2048, 2048]"""
    step = 1024
    preds = []
    for x in [0, step]:
        for y in [0, step]:
            preds.append(get_saliency(model, input_tensor[:, :, x:x+step, y:y+step],
                                     truth[x:x+step, y:y+step]))
    ul, ur, ll, lr = preds
    # Reconstruct from smaller portions
    up = np.concatenate((ul, ur), axis=1)
    down = np.concatenate((ll, lr), axis=1)
    sal = np.concatenate((up, down), axis=0)
    g_range = sal.max() - sal.min()
    sal /= g_range
    return sal

def get_crops(img):
    "assumes img is 2048 and target is 1024"
    return img[:1024, :1024], img[:1024, 1024:], img[1024:, :1024], img[1024:, 1024:]

def reconstruct(ul, ur, ll, lr):
    up = np.concatenate((ul, ur), axis=1)
    down = np.concatenate((ll, lr), axis=1)
    return np.concatenate((up, down), axis=0)

def _mini_predict_on_file(model, img, dem_img, flips=flip.FLIP_FULL, channels=4):
    """Handles logic of choosing which input tensor to create"""
    if channels == 4:
        input_tensor = make_input_tensor(img, dem_img)
    elif channels == 3:
        input_tensor = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    elif channels == 1:
        input_tensor = np.expand_dims(np.expand_dims(dem_img,axis=0),axis=0)
    with torch.no_grad():
        input_tensor = torch.Tensor(input_tensor)
        return predict(model, torch.autograd.Variable(input_tensor).cuda(), flips=flips)

def predict_on_file(model, fname, crop=True, flips=flip.FLIP_FULL, channels=4):
    """Predicts on 4 separate regions of the input file, then pieces them together if crop=True
    Averages predictions on flips of each image based upon flip parameter
    Chooses which input tensor to create depending upon number of channels

    channels == 4 -- RGB+DSM
    channels == 3 -- RGB
    channels == 1 -- DSM"""

    original_img = load_img(fname) / 255.0
    dsm = georaster.SingleBandRaster(fname.replace('RGB', 'DSM')).r
    dtm = georaster.SingleBandRaster(fname.replace('RGB', 'DTM')).r
    original_dem = (dsm - dtm) / 9.0
    pred_array = []
    if crop:
        for img, dem_img in zip(get_crops(original_img), get_crops(original_dem)):
            out = _mini_predict_on_file(model, img, dem_img, channels=channels, flips=flips)
            pred_array.append(np.transpose((out[0]*255).astype(np.uint8), (1, 2, 0)).squeeze())
        return reconstruct(*pred_array)
    else:
        out = _mini_predict_on_file(model, original_img, original_dem, flips=flips, channels=channels)
        return np.transpose((out[0]*255).astype(np.uint8), (1, 2, 0)).squeeze()

def plot_poly(contours, ax, coord_fn, color):
    """Plots a polygon where contours is a list of M polygons. Each polygon is an Nx2 array of 
    its vertices. Coord_fn is meant to be the coordinate function of a georaster image."""
    p = []
    for i, poly in enumerate(contours):
        # Avoid degenerate polygons
        if len(poly) < 3:
            continue
        pts = np.array(poly).squeeze()
        try:
            xs, ys = coord_fn(Xpixels=list(pts[:, 0]), Ypixels=list(pts[:, 1]))
        except IndexError as e:
            print("error on translating poly {}".format(i))
        p.append(Polygon(np.vstack([xs, ys]).T, facecolor='red'))
    col = PatchCollection(p)
    col.set_color(color)
    ax.add_collection(col)
    return ax
