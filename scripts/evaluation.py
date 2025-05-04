
# from visdom import Visdom
# viz = Visdom(port=8850)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from skimage.filters import threshold_otsu
import scipy.ndimage
from skimage.filters import threshold_otsu
from skimage.measure import regionprops, label
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def postprocess_evaluate(
        original_image, 
        reconstructed_image, 
        ground_truth_seg
    ):
    original_image = original_image.cpu().detach().numpy().squeeze()
    reconstructed_image = reconstructed_image.cpu().detach().numpy().squeeze()
    ground_truth_seg = ground_truth_seg.cpu().detach().numpy().squeeze()
    
    brain_mask = (np.abs(original_image).sum(axis=0) > 0).astype(float)
    
    print("original_image shape:", original_image.shape)
    print("reconstructed_image shape:", reconstructed_image.shape)
    print("ground_truth_seg shape:", ground_truth_seg.shape)
    
    print("original_image (min, max):", (original_image.min(), original_image.max()))
    print("reconstructed_image (min, max):", (reconstructed_image.min(), reconstructed_image.max()))
    
    if reconstructed_image.min() < 0:
        reconstructed_image = (reconstructed_image + 1.0) / 2.0
    
        print("reconstructed_image after normalization (min, max):", 
              (reconstructed_image.min(), reconstructed_image.max()))
    
    anomaly_map = 1 - np.abs(original_image - reconstructed_image).sum(axis=0)
    anomaly_map = anomaly_map * brain_mask
    
    anomaly_map_mf = scipy.ndimage.filters.median_filter(anomaly_map, size=5)
    
    print("anomaly_map (min, max):", (anomaly_map.min(), anomaly_map.max()))
    print("anomaly_map_mf (min, max):", (anomaly_map_mf.min(), anomaly_map_mf.max()))
    
    brain_values = anomaly_map_mf[brain_mask > 0]
    if len(brain_values) > 0:
        try:
            bestThresh1 = threshold_otsu(brain_values)
            bestDice, bestThresh2 = find_best_val(anomaly_map_mf.flatten(),
                                                ground_truth_seg.flatten().astype(bool), 
                                                val_range=(0, np.max(np.array(anomaly_map_mf))),
                                                max_steps=10, 
                                                step=0, 
                                                max_val=0, 
                                                max_point=0)
        except:
            # Fallback if above fails (e.g., all values are the same)
            bestThresh1 = np.mean(brain_values)
            bestThresh2 = bestThresh1
    else:
        bestThresh1 = 0
        bestThresh2 = 0
    
    print("bestThresh1, 2:", bestThresh1, bestThresh2)
    
    # Apply threshold to get binary anomaly map
    anomaly_thresholded_1 = np.where(anomaly_map_mf > bestThresh1, 1, 0)
    anomaly_thresholded_2 = np.where(anomaly_map_mf > bestThresh2, 1, 0)
    
    # Make sure anomaly is only detected within brain region
    anomaly_thresholded_1 = anomaly_thresholded_1 * brain_mask
    anomaly_thresholded_1 = filter_3d_connected_components(anomaly_thresholded_1)

    anomaly_thresholded_2 = anomaly_thresholded_2 * brain_mask
    anomaly_thresholded_2 = filter_3d_connected_components(anomaly_thresholded_2)
    
    # Calculate metrics
    auprc_score = average_precision_score(
        ground_truth_seg.flatten().astype(bool),
        anomaly_map_mf.flatten()
    )
    
    # You need to define the dice function or import it
    dice_score_1 = dice(ground_truth_seg, anomaly_thresholded_1)
    dice_score_2 = dice(ground_truth_seg, anomaly_thresholded_2)
    
    # Plot results
    titles = [
        "Original", "Reconstructed", "Anomaly Map", "Median Filtered",
        "Thresholded", "Ground Truth"
    ]
    
    images = [
        original_image.sum(axis=0).squeeze(), 
        reconstructed_image.sum(axis=0).squeeze(),
        anomaly_map,
        anomaly_map_mf,
        anomaly_thresholded_2,
        ground_truth_seg.squeeze()
    ]
    
    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    for ax, img, title in zip(axes, images, titles):
        if title not in ("Anomaly Map", "Median Filtered"):
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img, cmap='magma')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    
    return dice_score_1, dice_score_2, auprc_score, fig

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
    psum = np.sum(P.flatten())
    gsum = np.sum(G.flatten())
    pgsum = np.sum(np.multiply(P.flatten(), G.flatten()))
    score = (2 * pgsum) / (psum + gsum)
    return score

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