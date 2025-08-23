# Encoder Guided Diffusion Model for Unsupervised Anomaly Detection in Brain MRI Scans

We provide the Pytorch implementation of our novel approach Encoder Guided Diffusion Model for Unsupervised Anomaly Detection in Brain MRI Scans".

The implementation of Denoising Diffusion Probabilistic Models presented in the paper is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).

## Data

We evaluated our method on the [BRATS2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html).
A mini-example how the data needs to be stored can be found in the folder _data_. To train or evaluate on the dataset, set `--dataset brats` respectively.

## Usage

We set the flags as follows:

```
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim1000 --use_ddim True"
```

To train the diffusion model, run

```
python scripts/image_train.py --data_dir path_to_traindata --dataset brats_or_chexpert  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

The model will be saved in the _results_ folder.

For image-to-image translation to a healthy subject on the test set, run

```
python scripts/encoder_sample_known.py  --data_dir path_to_testdata --model_path ./results/model.pt --dataset brats --classifier_scale 0 --noise_level 200 $MODEL_FLAGS $DIFFUSION_FLAGS  $SAMPLE_FLAGS
```
