# *************************************nyuv2******************************************************************
# CUDA_VISIBLE_DEVICES=0 python /home/cyxiong/SGACNet1/feature_nyuv2.py \
#     --dataset nyuv2 \
#     --dataset_dir /home/cyxiong/SGACNet1/datasets/nyuv2 \
#     --ckpt_path /home/cyxiong/SGACNet1/results/nyuv2/49.44-R34-Our-bilinear-checkpoints_12_09_2022-23_30_15-974836/ckpt_epoch_475.pth

    # /home/cyxiong/SGACNet1/results/nyuv2/49.44-R34-Our-bilinear-checkpoints_12_09_2022-23_30_15-974836/ckpt_epoch_475.pth
# ********************************cityscapes_full******************************************
CUDA_VISIBLE_DEVICES=0 python /home/yzhang/SGACNet/feature_city.py \
    --dataset cityscapes-with-depth \
    --dataset_dir /home/yzhang/SGACNet/datasets/cityscapes \
    --ckpt_path /home/yzhang/SGACNet/results/cityscapes-with-depth/74.08/79.67-r34-SE+SPA147+BRU3K9-bili-checkpoints_28_07_2022-09_02_12-905806/ckpt_epoch_476.pth \
    --raw_depth \
    --he_init \
    --aug_scale_min 0.5 \
    --aug_scale_max 2.0 \
    --valid_full_res \
    --height 512 \
    --width 1024 \
    --batch_size 8 \
    --batch_size_valid 16 \
    --lr 1e-4 \
    --optimizer Adam \
    --class_weighting None \
    --encoder resnet34 \
    --encoder_block NonBottleneck1D \
    --nr_decoder_blocks 3 \
    --modality rgbd \
    --encoder_decoder_fusion add \
    --context_module appm-1-2-4-8 \
    --decoder_channels_mode decreasing \
    --fuse_depth_in_rgb_encoder SE-add \
    --upsampling bilinear
# ***********************************sunrgbd***************
# CUDA_VISIBLE_DEVICES=1 python /home/cyxiong/SGACNet1/feature_sunrgbd.py \
#     --dataset sunrgbd \
#     --dataset_dir /home/cyxiong/SGACNet1/datasets/sunrgbd \
#     --ckpt_path /home/cyxiong/SGACNet1/results/sunrgbd/47.81-SE+SPA147+BRU3K9+APPM+bili-checkpoints_30_07_2022-08_52_09-655286/ckpt_epoch_238.pth
