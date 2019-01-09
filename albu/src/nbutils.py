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
    g_range = abs_sal.max() - abs_sal.min()
    abs_sal /= g_range
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
    saliency = np.concatenate((up, down), axis=0)
    return saliency

def get_crops(img):
    "assumes img is 2048 and target is 1024"
    return img[:1024, :1024], img[:1024, 1024:], img[1024:, :1024], img[1024:, 1024:]

def reconstruct(ul, ur, ll, lr):
    up = np.concatenate((ul, ur), axis=1)
    down = np.concatenate((ll, lr), axis=1)
    return np.concatenate((up, down), axis=0)

def predict_on_file(model, fname, crop=True, flips=flip.FLIP_FULL, channels=4):
    """Predicts using RGB fname, loads in DTM+DSM data as well"""
    original_img = load_img(fname) / 255.0
    dsm = georaster.SingleBandRaster(fname.replace('RGB', 'DSM')).r
    dtm = georaster.SingleBandRaster(fname.replace('RGB', 'DTM')).r
    original_dem = (dsm - dtm) / 9.0
    pred_array = []
    if crop:
        for img, dem_img in zip(get_crops(original_img), get_crops(original_dem)):
            if channels == 4:
                input_tensor = make_input_tensor(img, dem_img)
            else:
                input_tensor = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
            with torch.no_grad():
                input_tensor = torch.Tensor(input_tensor)
                out = predict(model, torch.autograd.Variable(input_tensor).cuda(), flips=flips)

            pred_array.append(np.transpose((out[0]*255).astype(np.uint8), (1, 2, 0)).squeeze())
        return reconstruct(*pred_array)
    else:
        if channels == 4:
            input_tensor = make_input_tensor(original_img, original_dem)
        else:
            input_tensor = np.expand_dims(np.transpose(original_img, (2, 0, 1)), axis=0)
        with torch.no_grad():
            input_tensor = torch.Tensor(input_tensor)
            out = predict(model, torch.autograd.Variable(input_tensor).cuda(), flips=flips)
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
