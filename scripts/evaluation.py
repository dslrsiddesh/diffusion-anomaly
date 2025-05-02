
# from visdom import Visdom
# viz = Visdom(port=8850)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from skimage.filters import threshold_otsu
import scipy.ndimage
from skimage.measure import regionprops, label
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

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


def postprocess_evaluate(
        original_image, 
        reconstructed_image, 
        ground_truth_seg
    ):
    original_image = original_image.cpu().detach().numpy().squeeze()
    reconstructed_image = reconstructed_image.cpu().detach().numpy().squeeze()

    print("original_image (min, max): ", (original_image.min(),original_image.max()))
    print("reconstructed_image (min, max): ", (reconstructed_image.min(),reconstructed_image.max()))

    ground_truth_seg = ground_truth_seg.cpu().detach().numpy().squeeze()

    anomaly_map = np.abs(original_image - reconstructed_image)

    # print shapes of all first
    print("original_image shape: ", original_image.shape)
    print("reconstructed_image shape: ", reconstructed_image.shape)
    print("anomaly_map shape: ", anomaly_map.shape)
    print("ground_truth_seg shape: ", ground_truth_seg.shape)
    anomaly_map_mf = apply_3d_median_filter(anomaly_map, kernelsize=5)
    anomaly_map_mf = anomaly_map_mf.sum(axis=0)
    
    bestDice, bestThresh = find_best_val(
        anomaly_map_mf.flatten(),
        ground_truth_seg.flatten().astype(bool),
        val_range=(0, np.max(anomaly_map_mf)),
        max_steps=10
    )

    # anomaly_thresholded = anomaly_map_mf > bestThresh
    anomaly_thresholded = np.where(anomaly_map_mf > bestThresh, 1, 0)
    anomaly_thresholded_cc = filter_3d_connected_components(anomaly_thresholded)

    titles = [
        "Original", "Reconstructed", "Anomaly Map", "Median Filtered",
        "Thresholded", "Connected Components", "Ground Truth"
    ]

    # print shapes of all first
    # print("original_image shape: ", original_image.shape)
    # print("reconstructed_image shape: ", reconstructed_image.shape)
    # print("anomaly_map shape: ", anomaly_map.shape)
    # print("anomaly_map_mf shape: ", anomaly_map_mf.shape)
    # print("anomaly_thresholded shape: ", anomaly_thresholded.shape)
    # print("anomaly_thresholded_cc shape: ", anomaly_thresholded_cc.shape)
    # print("ground_truth_seg shape: ", ground_truth_seg.shape)

    images = [
        original_image.sum(axis=0).squeeze(), 
        reconstructed_image.sum(axis=0).squeeze(),
        anomaly_map.sum(axis=0).squeeze(),
        anomaly_map_mf,
        anomaly_thresholded,
        anomaly_thresholded_cc.squeeze(),
        ground_truth_seg.squeeze()
    ]

    fig, axes = plt.subplots(1, 7, figsize=(21, 3))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    # plt.show()

    auprc_score = average_precision_score(
        ground_truth_seg.flatten().astype(bool),
        anomaly_map_mf.flatten()
    )
    dice_score = dice(ground_truth_seg, anomaly_thresholded_cc)
    
    return dice_score, auprc_score, fig

def apply_3d_median_filter(volume, kernelsize=5):
    volume = scipy.ndimage.filters.median_filter(volume, (kernelsize, kernelsize, kernelsize))
    return volume

# From Zimmerer iterative algorithm for threshold search
def find_best_val(x, y, val_range=(0, 1), max_steps=4, step=0, max_val=0, max_point=0):  #x: Image , y: Label
    if step == max_steps:
        return max_val, max_point

    if val_range[0] == val_range[1]:
        val_range = (val_range[0], 1)

    bottom = val_range[0]
    top = val_range[1]
    center = bottom + (top - bottom) * 0.5

    q_bottom = bottom + (top - bottom) * 0.25
    q_top = bottom + (top - bottom) * 0.75
    val_bottom = dice(x > q_bottom, y)
    #print(str(np.mean(x>q_bottom)) + str(np.mean(y)))
    val_top = dice(x > q_top, y)
    #print(str(np.mean(x>q_top)) + str(np.mean(y)))
    #val_bottom = val_fn(x, y, q_bottom) # val_fn is the dice calculation dice(p, g)
    #val_top = val_fn(x, y, q_top)

    if val_bottom >= val_top:
        if val_bottom >= max_val:
            max_val = val_bottom
            max_point = q_bottom
        return find_best_val(x, y, val_range=(bottom, center), step=step + 1, max_steps=max_steps,
                             max_val=max_val, max_point=max_point)
    else:
        if val_top >= max_val:
            max_val = val_top
            max_point = q_top
        return find_best_val(x, y, val_range=(center, top), step=step + 1, max_steps=max_steps,
                             max_val=max_val,max_point=max_point)
def dice(P, G):
    # psum = np.sum(P.flatten())
    # gsum = np.sum(G.flatten())
    # pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    # score = (2 * pgsum) / (psum + gsum)
    # return score
    P = (P > 0).float()
    G = (G > 0).float()
    intersection = (P * G).sum()
    return 2. * intersection / (P.sum() + G.sum())

def filter_3d_connected_components(volume):
    sz = None
    if volume.ndim > 3:
        sz = volume.shape
        volume = np.reshape(volume, [sz[0] * sz[1], sz[2], sz[3]])

    cc_volume = label(volume, connectivity=2)
    props = regionprops(cc_volume)
    for prop in props:
        if prop['filled_area'] <= 7:
            volume[cc_volume == prop['label']] = 0

    if sz is not None:
        volume = np.reshape(volume, [sz[0], sz[1], sz[2], sz[3]])
    return volume