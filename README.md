# SGACNet:Spatial-information Guided Adaptive Context-aware Network for Efficient RGB-D Semantic Segmentation
This repository contains the code to our paper "Spatial-information Guided Adaptive Context-aware Network for Efficient RGB-D Semantic Segmentation" ([arXiv]()).

## Model Zoo
### Validation on NYUDv2

|           Backbone         |   PixAcc    |    mAcc     |    mIoU    | Input |  Model | 
| :------------------------: | :---------: | :---------: | :--------: | :---: |  :---: | 
|      **ResNet18-NBt1D**    |    74.6     |    61.8     |    48.2    | RGBD  | [model]() |
|      **ResNet34-NBt1D**    |    75.6     |    62.7     |    49.4    | RGBD  | [model]() | 

### Validation on SUN RGB-D

|           Backbone         |   PixAcc    |    mAcc     |    mIoU    | Input |  Model | 
| :------------------------: | :---------: | :---------: | :--------: | :---: |  :---: | 
|      **ResNet18-NBt1D**    |    81.0     |    57.8     |    46.5    | RGBD  | [model]() |
|      **ResNet34-NBt1D**    |    81.2     |    60.8     |    47.8    | RGBD  | [model]() | 

### Validation on Cityscapes

|           Backbone         |   Params    |  mIoU.half  |  mIoU.full | Input |  Model | 
| :------------------------: | :---------: | :---------: | :--------: | :---: |  :---: | 
|      **ResNet18-NBt1D**    |    22.1     |    73.3     |    78.7    | RGBD  | [model]() |
|      **ResNet34-NBt1D**    |    35.6     |    74.1     |    79.7    | RGBD  | [model]() | 





## Citations
>Yang Zhang, Chenyun Xiong, Junjie Liu, Xuhui Ye, and Guodong Sun. Spatial-information Guided Adaptive Context-aware Network for Efficient RGBD Semantic Segmentation[J]. IEEE Sensors Journal, 2023.
## Installation
### 1.Clone repository:
Please be navigate to the cloned directory.
```
git clone --recursive https://github.com/CyunXiong/SGACNet.git
cd /path/to/this/repository
```
### 2.Create conda environment and install all dependencies:
Note we are using python 3.7+. Torch 1.3.1 and torchvision 0.4.2
```
conda env create -f rgbd_segmentation.yaml
conda activate SGACNet
```
### 3. Data Preparation
We trained our networks on [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), [SUNRGB-D](https://rgbd.cs.princeton.edu/), and [Cityscapes](https://www.cityscapes-dataset.com/). And they are stored in ```<dir>/datasets```.
### 4.Download pre-trained ImageNet models
Pre-trained ImageNet models can be downloaded for our selected [ResNet34-NBt1D]() backbones on the above datasets. Stored in ```<dir>/trained_models/imagenet```.
>* Note that we reported the inference time for all datasets in our paper.
## Training
Use ```main.py``` to train SGACNet on NYUv2, SUNRGB-D and Cityscapes. Otherwise, you can use imagenet_pretraining.py to create your own pretrained weights.

Example: 

* Train our SGACNet on NYUv2: 
> Run ```sh train_nyu.sh(train_sunrgbd.sh/train_cityscapes.sh)```.
```
#train_nyu.sh
python train.py \
    --dataset nyuv2 \
    --dataset_dir ./datasets/nyuv2 \
    --pretrained_dir ./trained_models/imagenet \
    --results_dir ./results \
    --height 480 \
    --width 640 \
    --batch_size 16 \
    --batch_size_valid 24 \
    --lr 0.01 \
    --optimizer SGD \
    --class_weighting median_frequency \
    --encoder resnet34 \
    --encoder_block NonBottleneck1D \
    --nr_decoder_blocks 3 \
    --modality rgbd \
    --encoder_decoder_fusion add \
    --context_module ppm \
    --decoder_channels_mode decreasing \
    --fuse_depth_in_rgb_encoder SE-add \
    --upsampling learned-3x3-zeropad
```
> Note that the some parameters are different in Cityscapes.
## Evaluation
To reproduce the metrics reported in our paper, run ```sh eval.sh ```.
Example: 

* To evaluate our SGACNet trained on NYUv2, use:
```
# eval_nyuv.sh
python eval.py \
    --dataset nyuv2 \
    --dataset_dir ./datasets/nyuv2 \
    --ckpt_path ./results/nyuv2/ckpt_epoch_best.pth
   ```
 > Evaluation on SUN RGB-D is similar to NYUv2.
 
* To evaluate our SGACNet trained on Cityscapes, use:
 ```
# eval_city.sh
  python eval.py \
    --dataset cityscapes-with-depth \
    --dataset_dir ./datasets/cityscapes \
    --ckpt_path ./trained_models/cityscapes/ckpt_epoch_best.pth \
    --height 512 \
    --width 1024 \
    --raw_depth \
    --context_module appm-1-2-4-8 \
    --valid_full_res 
  ```
## Time Inference 
We timed the inference on a single NVIDIA RTX 3090Ti with CUDA 11.7.
Example: 
* To reproduce the timings of our SGACNet trained on NYUv2, run eval_nyuv.sh:
 ```
 python3 ./inference_time_whole_model.py \
    --dataset nyuv2 \
    --no_time_pytorch \
    --no_time_tensorrt \
    --trt_floatx 16 \
    --plot_timing \
    --plot_outputs \
    --export_outputs
 ```







