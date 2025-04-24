
# from visdom import Visdom
# viz = Visdom(port=8850)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from skimage.filters import threshold_otsu

def normalize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def evaluate(diff_array, gt_mask):
    otsu_thresh = threshold_otsu(diff_array)
    otsu_mask = th.where(th.tensor(diff_array) > otsu_thresh, 1, 0)  
    # viz.image(normalize(otsu_mask), opts=dict(caption="otsu_mask"))

    Labelmask_GT = th.where(th.tensor(gt_mask) > 0, 1, 0)

    pixel_wise_cls = normalize(np.array(th.tensor(diff_array).view(1, -1))[0, :])
    pixel_wise_gt = normalize(np.array(th.tensor(Labelmask_GT).view(1, -1))[0, :])

    otsu_dice=dice_score(otsu_mask.cpu(), Labelmask_GT.cpu()) 
    auroc = roc_auc_score(pixel_wise_gt, pixel_wise_cls)

    return otsu_dice.item(), auroc, normalize(otsu_mask)
