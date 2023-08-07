# CUDA_VISIBLE_DEVICES=0  python /home/yzhang/SGACNet1/eval.py \
#     --dataset nyuv2 \
#     --dataset_dir /home/yzhang/SGACNet1/datasets/nyuv2 \
#     --ckpt_path /home/yzhang/SGACNet1/results/nyuv2/49.13-re-r34-checkpoints_31_07_2022-14_43_15-846351/ckpt_epoch_305.pth

#* nyuv2
# 49.44,R34,Our /home/yzhang/SGACNet1/results/nyuv2/49.44-R34-Our-bilinear-checkpoints_12_09_2022-23_30_15-974836/ckpt_epoch_475.pth
# 49.13,R34, base /home/yzhang/SGACNet1/results/nyuv2/49.13-re-r34-checkpoints_31_07_2022-14_43_15-846351/ckpt_epoch_305.pth
# 48.19,R18,Our /home/yzhang/SGACNet1/results/nyuv2/48.19-R18-Our-L3X3-SE+SPA147+AXIU-BRU3K9+APPM-bilinear-checkpoints_14_07_2022-10_10_16-699408/ckpt_epoch_212.pth
# 
# ****************************************
CUDA_VISIBLE_DEVICES=0  python /home/yzhang/SGACNet/eval.py \
    --dataset sunrgbd \
    --dataset_dir /home/yzhang/SGACNet/datasets/sunrgbd \
    --ckpt_path /home/yzhang/SGACNet/results/sunrgbd/47.46,R34-ORI-checkpoints_18_07_2022-10_51_39-041621/ckpt_epoch_498.pth

#* sunrgbd
# /home/yzhang/SGACNet1/results/sunrgbd/47.81-SE+SPA147+BRU3K9+APPM+bili-checkpoints_30_07_2022-08_52_09-655286/ckpt_epoch_238.pth
# 47.46,Our,r18   /home/yzhang/SGACNet1/results/sunrgbd/46.46,r18-SE+SPA147+BRU3K9+APPM+bili-checkpoints_01_08_2022-14_17_16-294953/ckpt_epoch_400.pth
# 47.46,R34,base /home/yzhang/SGACNet1/results/sunrgbd/47.46,R34-ORI-checkpoints_18_07_2022-10_51_39-041621/ckpt_epoch_498.pth
# *****************************
# CUDA_VISIBLE_DEVICES=0  python /home/yzhang/SGACNet1/eval.py \
#     --dataset cityscapes-with-depth \
#     --dataset_dir /home/yzhang/SGACNet1/datasets/cityscapes\
#     --raw_depth \
#     --he_init \
#     --aug_scale_min 0.5 \
#     --aug_scale_max 2.0 \
#     --valid_full_res \
#     --height 512 \
#     --width 1024 \
#     --modality rgb \
#     --optimizer Adam \
#     --ckpt_path /home/yzhang/SGACNet1/results/cityscapes-with-depth/70.8,77.65-resnet34-ori-rgb-checkpoints_18_07_2022-15_53_09-586021/ckpt_epoch_391.pth
 # --valid_full_res \   
#*R34,Our,74.08,rgbd,half /home/yzhang/SGACNet1/results/cityscapes-with-depth/74.08-79.7/79.67-r34-SE+SPA147+BRU3K9-bili-checkpoints_28_07_2022-09_02_12-905806/ckpt_epoch_403.pth
#*R34,Our,79.7,rgbd /home/yzhang/SGACNet1/results/cityscapes-with-depth/74.08-79.7/79.67-r34-SE+SPA147+BRU3K9-bili-checkpoints_28_07_2022-09_02_12-905806/ckpt_epoch_476.pth
# *R34,Our,72.3,rgb,half /home/yzhang/SGACNet1/results/cityscapes-with-depth/72.3,78.5-r34-rgb-SE+SPA147+AXIU-BRU3K9+APPM-bilinear-checkpoints_20_07_2022-17_22_13-719991/ckpt_epoch_380.pth
# *R34,Our,78.5,rgb, /home/yzhang/SGACNet1/results/cityscapes-with-depth/72.3,78.5-r34-rgb-SE+SPA147+AXIU-BRU3K9+APPM-bilinear-checkpoints_20_07_2022-17_22_13-719991/ckpt_epoch_380.pth
# *R18,Our,73.3,rgbd,half /home/yzhang/SGACNet1/results/cityscapes-with-depth/73.3_78.7_Our-R18+rgbd+bili_checkpoints_11_10_2022-15_40_04-708927/ckpt_epoch_463.pth
# *R18,Our,78.66,rgbd /home/yzhang/SGACNet1/results/cityscapes-with-depth/73.3_78.7_Our-R18+rgbd+bili_checkpoints_11_10_2022-15_40_04-708927/ckpt_epoch_463.pth
# *R18,Our,77.50,rgb /home/yzhang/SGACNet1/results/cityscapes-with-depth/72.20_77.50_our_rgb-R18+bili+checkpoints_13_10_2022-09_35_27-537066/ckpt_epoch_496.pth
#*R34,b,79.79,rgbd /home/yzhang/SGACNet1/results/cityscapes-with-depth/79.6,79.79-r34-ori-rgbd-checkpoints_18_07_2022-10_45_30-692751/ckpt_epoch_482.pth
#*R34,b,70.8,77.65,rgb /home/yzhang/SGACNet1/results/cityscapes-with-depth/70.8,77.65-resnet34-ori-rgb-checkpoints_18_07_2022-15_53_09-586021/ckpt_epoch_391.pth