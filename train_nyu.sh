CUDA_VISIBLE_DEVICES=0 python /home/yzhang/SGACNet/train.py \
    --dataset nyuv2 \
    --dataset_dir /home/yzhang/SGACNet/datasets/nyuv2 \
    --pretrained_dir /home/yzhang/SGACNet/trained_models/imagenet \
    --results_dir /home/yzhang/SGACNet/results \
    --height 480 \
    --width 640 \
    --batch_size 16 \
    --batch_size_valid 24 \
    --lr 0.01 \
    --optimizer SGD \
    --class_weighting median_frequency \
    --encoder resnet18 \
    --encoder_block NonBottleneck1D \
    --nr_decoder_blocks 3 \
    --modality rgbd \
    --encoder_decoder_fusion add \
    --context_module appm \
    --decoder_channels_mode decreasing \
    --fuse_depth_in_rgb_encoder SE-add \
    --upsampling bilinear

 #* nyuv2
# 49.44,R34,Our /home/yzhang/SGACNet1/results/nyuv2/49.44-R34-Our-bilinear-checkpoints_12_09_2022-23_30_15-974836/ckpt_epoch_475.pth
# 49.13,R34, base /home/yzhang/SGACNet1/results/nyuv2/49.13-re-r34-checkpoints_31_07_2022-14_43_15-846351/ckpt_epoch_305.pth


#* sunrgbd
# /home/yzhang/SGACNet1/results/sunrgbd/47.81-SE+SPA147+BRU3K9+APPM+bili-checkpoints_30_07_2022-08_52_09-655286/ckpt_epoch_238.pth
