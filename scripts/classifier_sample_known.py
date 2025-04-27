"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from datetime import datetime
# from visdom import Visdom
# viz = Visdom(port=8850)
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
from guided_diffusion.train_util import get_device_details
from scripts.evaluation import evaluate
import wandb

def visualize(img):
    """Returns min-max normalized image."""
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def main():    
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    # set current date and time as dir name for logs
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    project = "CFG_eval_100"
    wandb.init(project=project)
    # Access hyperparameters from wandb.config
    config = wandb.config

    dist_util.setup_dist()
    
    # logger.configure(dir=f'./{project}/sweep/')

    get_device_details()

    logger.log("-----"*10)
    logger.log("Parsed arguments:")
    for arg, value in vars(args).items():
        logger.log(f"{arg}: {value}")
    logger.log("-----"*10)

    logger.log("loading dataset...")
    if args.dataset=='brats':
        ds = BRATSDataset(config.data_dir, test_flag=True)
        datal = th.utils.data.DataLoader(
                                        ds,
                                        batch_size=args.batch_size,
                                        shuffle=False
                                        )
        logger.log(f"found {len(ds)} test samples of brats")

    elif args.dataset=='chexpert':
        data = load_data(
                        data_dir=config.data_dir,
                        batch_size=args.batch_size,
                        image_size=args.image_size,
                        class_cond=True,
                        )
        datal = iter(data)
   

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
                                                **args_to_dict(args, model_and_diffusion_defaults().keys())
                                                    )
    
    model.load_state_dict(
                        dist_util.load_state_dict(config.model_path,map_location="cpu")
                        )
    model.to(dist_util.dev())
    print("loaded model")

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # logger.log("loading classifier...")
    # classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    # classifier.load_state_dict(
    #                             dist_util.load_state_dict(args.classifier_path)
    #                             )
    # classifier.to(dist_util.dev())
    # print('loaded classifier')

    # if args.classifier_use_fp16:
    #     classifier.convert_to_fp16()
    # classifier.eval()

    p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
    # p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
    logger.log(f"model parameters: {p1}")
    # logger.log(f"classifier parameters: {p2}")


    # def cond_fn(x, t,  y=None):
    #     """Condition Guidance Function"""
    #     assert y is not None
    #     with th.enable_grad():
    #         x_in = x.detach().requires_grad_(True)
    #         logits = classifier(x_in, t)
    #         log_probs = F.log_softmax(logits, dim=-1)
    #         selected = log_probs[range(len(logits)), y.view(-1)]
    #         a=th.autograd.grad(selected.sum(), x_in)[0]
    #         return  a, a * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    dice_scores = []
    auroc_scores = []

    for n, img in enumerate(datal):

        model_kwargs = {}
     #   img = next(data)  # should return an image from the dataloader "data"
        print('img', img[0].shape, img[1])
        if args.dataset=='brats':
          Labelmask = th.where(img[3] > 0, 1, 0)
          number=img[4][0]
          if img[2]==0:
              continue    #take only diseased images as input
              
          #viz.image(visualize(img[0][0, 0, ...]), opts=dict(caption="img input 0"))
          #viz.image(visualize(img[0][0, 1, ...]), opts=dict(caption="img input 1"))
          #viz.image(visualize(img[0][0, 2, ...]), opts=dict(caption="img input 2"))
          #viz.image(visualize(img[0][0, 3, ...]), opts=dict(caption="img input 3"))
          gt_seg_padded = np.pad(th.squeeze(img[3][0, ...]), ((8, 8), (8, 8)), mode='constant', constant_values = 0)
          #viz.image(visualize(gt_seg_padded), opts=dict(caption="ground truth"))

        #   wandb.log({f"inputs_{str(img[4])}": [wandb.Image(visualize(img[0][0, 0, ...]), caption="img input 0"),
        #                         wandb.Image(visualize(img[0][0, 1, ...]), caption="img input 1"),
        #                         wandb.Image(visualize(img[0][0, 2, ...]), caption="img input 2"),
        #                         wandb.Image(visualize(img[0][0, 3, ...]), caption="img input 3"),
        #                         wandb.Image(visualize(img[3][0, ...]), caption="ground truth")]
        #              })
        else:
          #viz.image(visualize(img[0][0, ...]), opts=dict(caption="img input"))
          print('img1', img[1])
          number=img[1]["path"]
          print('number', number)

        if args.class_cond:
            classes = th.randint(
                                low=0, 
                                high=1, 
                                size=(args.batch_size,), 
                                device=dist_util.dev()
                                )
            model_kwargs["y"] = classes
            print('Conditioning for class:::', model_kwargs["y"])
            print(f"CFG Scale:::{config.classifier_free_guidance_scale}")
        sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                    )
        print('sample fn = ', sample_fn)

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        start.record()
        sample, x_noisy, org = sample_fn(
                                        model_fn,
                                        (args.batch_size, 4, args.image_size, args.image_size),
                                        img, 
                                        org=img,
                                        clip_denoised=args.clip_denoised,
                                        model_kwargs=model_kwargs,
                                        cond_fn=None,
                                        device=dist_util.dev(),
                                        noise_level=config.noise_level,
                                        classifier_free_guidance_scale=config.classifier_free_guidance_scale, # Add CFG scale
                                    )
        end.record()
        th.cuda.synchronize()
        th.cuda.current_stream().synchronize()

        print('time taken:', start.elapsed_time(end))

        if args.dataset=='brats':
            #viz.image(visualize(sample[0,0, ...]), opts=dict(caption="sampled output0"))
            #viz.image(visualize(sample[0,1, ...]), opts=dict(caption="sampled output1"))
            #viz.image(visualize(sample[0,2, ...]), opts=dict(caption="sampled output2"))
            #viz.image(visualize(sample[0,3, ...]), opts=dict(caption="sampled output3"))
            difftot=abs(org[0, :4,...]-sample[0, ...]).sum(dim=0)
            #viz.heatmap(visualize(difftot), opts=dict(caption="difftot"))

            # wandb.log({f"sampled_outputs_{str(img[4])}": [wandb.Image(visualize(sample[0, 0, ...]), caption="sampled output0"),
            #                                       wandb.Image(visualize(sample[0, 1, ...]), caption="sampled output1"),
            #                                       wandb.Image(visualize(sample[0, 2, ...]), caption="sampled output2"),
            #                                       wandb.Image(visualize(sample[0, 3, ...]), caption="sampled output3")]
            #             })
          
            # plt.figure()
            # sns.heatmap(np.array(difftot.cpu()), cmap='viridis', annot=False)
            # sns.heatmap(visualize(th.squeeze(img[3][0, ...]).cpu()), cmap='viridis', annot=False)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            sns.heatmap(np.array(difftot.cpu()), ax=axes[0], cbar=False, cmap='viridis', annot=False) 
            sns.heatmap(visualize(gt_seg_padded), ax=axes[1], cbar=False, cmap='viridis', annot=False) 
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            
            # wandb.log({
            #         f"anomaly_map_{str(img[4])}": wandb.Image(fig), #caption="difftot"
            #                             #    wandb.Image(visualize(img[3][0, ...]), caption=f"ground truth_{n+1}")]
            #     })
            #viz.matplot(plt, opts=dict(caption=f"anomaly_map_{str(img[4])}"))
            plt.clf()

            otsu_dice, auroc, otsu_mask = evaluate(np.array(difftot.cpu()), gt_seg_padded)

            fig, axes = plt.subplots(1, 3, figsize=(12, 6))
            sns.heatmap(np.array(difftot.cpu()), ax=axes[0], cbar=False, cmap='viridis', annot=False) 
            sns.heatmap(np.array(otsu_mask.cpu()), ax=axes[1], cbar=False, cmap='viridis', annot=False) 
            sns.heatmap(visualize(gt_seg_padded), ax=axes[2], cbar=False, cmap='viridis', annot=False) 
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            wandb.log({
                    f"anomaly_map_th_{str(img[4])}": wandb.Image(fig, caption=f"{otsu_dice=:.3f}, {auroc=:.3f}"), 
                })
            plt.clf()
            dice_scores.append(otsu_dice)
            auroc_scores.append(auroc)
            print(f"Sample: {str(img[4])}, {otsu_dice=}, {auroc=}")
            
        elif args.dataset=='chexpert':
          #viz.image(visualize(sample[0, ...]), opts=dict(caption="sampled output"+str(name)))
          diff=abs(visualize(org[0, 0,...])-visualize(sample[0,0, ...]))
          diff=np.array(diff.cpu())
          #viz.heatmap(np.flipud(diff), opts=dict(caption="diff"))
        

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

    wandb.log({"mean_dice_score": np.mean(dice_scores), "mean_auroc_score": np.mean(auroc_scores)})
    print(f"{len(dice_scores)} samples tested, Mean dice score: {np.mean(dice_scores)}, Mean AUROC: {np.mean(auroc_scores)}")  
    print(dice_scores)
    print(auroc_scores)

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
                    data_dir="",
                    clip_denoised=True,
                    num_samples=1,
                    batch_size=1,
                    use_ddim=True,
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

if __name__ == "__main__":
    main()

