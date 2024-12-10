import os
import nibabel as nib
import numpy as np
import torch as th
import argparse
from skimage.filters import threshold_otsu
from sklearn.metrics import roc_auc_score
# from visdom import Visdom

# Visualization utility
# viz = Visdom(port=8850)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img

def dice_score(pred, targs):
    pred = (pred > 0).float()
    return 2. * (pred * targs).sum() / (pred + targs).sum()

def process_and_evaluate(test_data_dir, sampled_data_dir):

    results = []
    for dir_name in os.listdir(test_data_dir):
        dir_path = os.path.join(test_data_dir, dir_name)
        if os.path.isdir(dir_path):
            flair_path = None
            t1_path = None
            t1ce_path = None
            t2_path = None
            seg_path = None
            
            # Extract paths for all modalities and ground truth (seg)
            for file_name in os.listdir(dir_path):
                if "flair" in file_name:
                    flair_path = os.path.join(dir_path, file_name)
                elif "t1ce" in file_name:
                    t1ce_path = os.path.join(dir_path, file_name)
                elif "t1" in file_name and "t1ce" not in file_name:
                    t1_path = os.path.join(dir_path, file_name)
                elif "t2" in file_name:
                    t2_path = os.path.join(dir_path, file_name)
                elif "seg" in file_name:
                    seg_path = os.path.join(dir_path, file_name)

            print(f"Processing {flair_path.split('/')[-1].replace('_flair_', '_X_')} files...")
            # print("Flair path: ", flair_path)
            # print("T1 path: ", t1_path)
            # print("T1ce path: ", t1ce_path)
            # print("T2 path: ", t2_path)
            # Check that all required files are found
            if not all([flair_path, t1_path, t1ce_path, t2_path, seg_path]):
                print(f"Missing required modalities in {dir_name}, skipping...")
                continue

            # Process corresponding sampled data
            flair_sampled_path = os.path.join(sampled_data_dir, f"{flair_path.split('/')[-1].split('.')[0]}_sampled.nii.gz")
            t1_sampled_path = os.path.join(sampled_data_dir, f"{t1_path.split('/')[-1].split('.')[0]}_sampled.nii.gz")
            t1ce_sampled_path = os.path.join(sampled_data_dir, f"{t1ce_path.split('/')[-1].split('.')[0]}_sampled.nii.gz")
            t2_sampled_path = os.path.join(sampled_data_dir, f"{t2_path.split('/')[-1].split('.')[0]}_sampled.nii.gz")

            # Load the .nii.gz files
            try:
                flair = nib.load(flair_path).get_fdata()
                t1 = nib.load(t1_path).get_fdata()
                t1ce = nib.load(t1ce_path).get_fdata()
                t2 = nib.load(t2_path).get_fdata()
                seg = nib.load(seg_path).get_fdata()
                
                flair_sampled = nib.load(flair_sampled_path).get_fdata()
                t1_sampled = nib.load(t1_sampled_path).get_fdata()
                t1ce_sampled = nib.load(t1ce_sampled_path).get_fdata()
                t2_sampled = nib.load(t2_sampled_path).get_fdata()
            except FileNotFoundError as e:
                print(f"Missing sampled file for {flair_path.split('_flair_').join('_')}, skipping...")
                continue

            # Compute the absolute differences for anomaly map
            anomaly_map = (
                np.abs(flair - flair_sampled) +
                np.abs(t1 - t1_sampled) +
                np.abs(t1ce - t1ce_sampled) +
                np.abs(t2 - t2_sampled)
            )

            # Compute Otsu threshold
            threshold = threshold_otsu(anomaly_map)
            binary_mask = th.where(th.tensor(anomaly_map) > threshold, 1, 0)

            # Load ground truth segmentation
            ground_truth = th.tensor((seg > 0).astype(np.float32))

            # Calculate Dice score
            dice = dice_score(binary_mask, ground_truth)

            # Visualize results
            # viz.image(visualize(binary_mask.numpy()[0, ...]), opts=dict(caption=f"Mask "))

            # Calculate Dice and AUROC scores
            DSC = dice_score(binary_mask.cpu(), ground_truth.cpu())

            results.append(DSC)

    if not results:
        return None
    else:
        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection scores.")
    parser.add_argument("--test_data_dir", type=str, required=True, help="Path to the test data directory")
    parser.add_argument("--sampled_data_dir", type=str, required=True, help="Path to the sampled data directory")
    args = parser.parse_args()

    test_data_dir = args.test_data_dir
    sampled_data_dir = args.sampled_data_dir

    results =  process_and_evaluate(test_data_dir, sampled_data_dir)

    # Print summary
    print("\nSummary (All the dice scores):")
    for DSC in results:
        print(f"Dice Score = {DSC:.4f}")


if __name__ == "__main__":
    main()
