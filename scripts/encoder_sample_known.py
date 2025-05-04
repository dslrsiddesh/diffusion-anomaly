"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
import torch.nn.functional as F
import numpy as np
import torch as th
import torch.distributed as dist
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from scripts.evaluation import postprocess_evaluate
from guided_diffusion.encoder import AEEncoder
from timm.utils import ModelEmaV3
import wandb
wandb.login(key="b3990523b6efe11b984fad8a8e64911f78399735")

project = "DDPM_Simplex_eval_re"
wandb.init(project=project, reinit=True)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def main():
    print('------------------------ in main ------------------------')
    device = dist_util.dev()
    print("Device:", device)

    encoder_checkpoint = th.load("/kaggle/input/ae-2channel/ckpt_2channel_epoch25.pth", map_location="cpu")
    base_encoder = AEEncoder(in_channels=2)

    ema_encoder_model = ModelEmaV3(base_encoder, decay=0.999)
    ema_encoder_model.load_state_dict(encoder_checkpoint['ema_encoder'])

    ema_encoder = ema_encoder_model.module
    ema_encoder = ema_encoder.to(dist_util.dev())
    ema_encoder.eval()
    print("---------------------------ema_encoder loaded and set to eval mode---------------------------")

    args = create_argparser().parse_args()
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print("------------------------- before brats ------------------------")
    if args.dataset=='brats':
      ds = BRATSDataset(args.data_dir, test_flag=True)
      datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)
    
    print("----------------- after brats ------------------------")
   
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # set current date and time as dir name for logs
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    print('pmodel', p1)

    def model_fn(x, t, y=None, encoder_features=None):
        if encoder_features is not None:
            print(f"------ Encoder features received with starting shape: {encoder_features[256].shape}------")
        return model(x, t, y if args.class_cond else None, encoder_features=encoder_features)

    logger.log("sampling...")
    all_numbers = []
    all_images = []
    all_labels = []
    dice_scores_nor = []
    dice_scores_zimmer = []
    auprc_scores = []

    for n, img in enumerate(datal):
        model_kwargs = {}
        print('img', img[0].shape, img[1])
        if args.dataset=='brats':
            Labelmask = th.where(img[3] > 0, 1, 0)
            b, c, h, w = Labelmask.shape
            pad_h = 256 - h
            pad_w = 256 - w
            
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            Labelmask = F.pad(Labelmask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            print(f"Labelmask: {Labelmask.shape}")
            assert Labelmask.shape[-1] == 256

            if isinstance(img[4], list):
                numbers = img[4]
            else:
                numbers = [img[4][i] for i in range(img[0].shape[0])]
                
            if img[2].sum() == 0:
                continue
        else:
            assert args.dataset=='brats'

        # sample_fn = (
        #     diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
        # )
        sample_fn = (
            diffusion.p_sample_loop_known
        )
        print('sample fn = ', sample_fn)

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        start.record()

        if ema_encoder is not None:
            print("===== Encoder features extraction in model_fn in encoder_sample_known.py =====")
            with th.no_grad():
                org_img = img[0].to(dist_util.dev())
                print("org_img shape sending to encoder: ", org_img.shape)
                encoder_features = ema_encoder(org_img)
                print('encoder_feats: ', encoder_features.keys())
        else:
            encoder_features = None
            print("===== No encoder features extraction because of the model=None =====")
        try: 
            sample, x_noisy, org = sample_fn(
                                            model_fn,
                                            (args.batch_size, 2, args.image_size, args.image_size),
                                            img, 
                                            org=img,
                                            clip_denoised=args.clip_denoised,
                                            model_kwargs=model_kwargs,
                                            cond_fn=None,
                                            device=dist_util.dev(),
                                            noise_level=args.noise_level,
                                            encoder_features=encoder_features,
                                        )
            end.record()
            th.cuda.synchronize()
            th.cuda.current_stream().synchronize()
            print('time taken:', start.elapsed_time(end))
        except RuntimeError:
            continue

        # if args.dataset=='brats':
        #     pp_dice, pp_auprc, pp_figure = postprocess_evaluate(org, sample, Labelmask)

        #     wandb.log({
        #             f"anomaly_map_th_{str(img[4])}": wandb.Image(pp_figure, caption=f"Dice Score{pp_dice:.3f}, {pp_auprc:.3f}"), 
        #         })
        #     plt.clf()
        #     all_numbers.append(number)
        #     dice_scores.append(pp_dice)
        #     auprc_scores.append(pp_auprc)
        #     print(f"Sample: {str(img[4])}, {pp_dice=}, {pp_auprc=}")

        # print("<<<<<<<<<<<<<<<<<<<<<<Sampling completed>>>>>>>>>>>>>>>>>>>>>>>")
        if args.dataset=='brats':
            for batch_idx in range(args.batch_size):
                if hasattr(img[2], 'shape') and img[2].shape[0] > 1 and img[2][batch_idx] == 0:
                    continue
                    
                single_org = extract_sample_from_batch(org, batch_idx)
                single_sample = sample[batch_idx:batch_idx+1]  
                single_labelmask = Labelmask[batch_idx:batch_idx+1]
                current_number = numbers[batch_idx]
                
                pp_dice_nor, pp_dice_zimmer, pp_auprc, pp_figure = postprocess_evaluate(single_org, single_sample, single_labelmask)

                wandb.log({
                        f"anomaly_map_th_{str(current_number)}": wandb.Image(pp_figure, caption=f"Dice Score{pp_dice_nor:.3f}, {pp_dice_zimmer:.3f}, {pp_auprc:.3f}"), 
                    })
                plt.clf()
                all_numbers.append(current_number)
                dice_scores_nor.append(pp_dice_nor)
                dice_scores_zimmer.append(pp_dice_zimmer)
                auprc_scores.append(pp_auprc)
                print(f"Sample: {str(current_number)}, {pp_dice_nor=}, {pp_dice_zimmer=}, {pp_auprc=}")

    wandb.log({"mean_dice_score_normal": np.mean(dice_scores_nor), "mean_dice_score_zimmer": np.mean(dice_scores_zimmer), "mean_auprc_score": np.mean(auprc_scores)})
    print(f"{len(dice_scores_nor)} samples tested, Mean dice score: {np.mean(dice_scores_nor)}, {np.mean(dice_scores_zimmer)}, Mean AUPRC: {np.mean(auprc_scores)}")
    print(dice_scores_nor)
    print(dice_scores_zimmer)
    print(auprc_scores)

    import pandas as pd
    results_df = pd.DataFrame({
        'number': all_numbers,
        'dice_score_normal': dice_scores_nor,
        'dice_score_zimmer': dice_scores_zimmer,
        'auprc': auprc_scores
    })
    results_df.to_csv(f"./results_{date_time_str}.csv", index=False)

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

    dist.barrier()
    logger.log("sampling complete")
    wandb.finish()


def create_argparser():
    defaults = dict(
        output_dir="sampled",
        data_dir="",
        clip_denoised="zero",
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset='brats',
        classifier_free_guidance_scale=2.0,
    )
    defaults.update(model_and_diffusion_defaults())
    # defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def extract_sample_from_batch(batch_data, idx):
    if isinstance(batch_data, th.Tensor):
        return batch_data[idx:idx+1]
    elif isinstance(batch_data, tuple) or isinstance(batch_data, list):
        return [extract_sample_from_batch(item, idx) if isinstance(item, (th.Tensor, tuple, list)) else item for item in batch_data]
    else:
        return batch_data

if __name__ == "__main__":
    main()