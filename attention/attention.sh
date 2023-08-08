# *************************************nyuv2******************************************************************
# CUDA_VISIBLE_DEVICES=0 python /home/cyxiong/SGACNet1/attenton_nyuv2.py \
    # --dataset nyuv2 \
    # --dataset_dir /home/cyxiong/SGACNet1/datasets/nyuv2 \
    # --ckpt_path /home/cyxiong/SGACNet1/results/nyuv2/49.44-R34-Our-bilinear-checkpoints_12_09_2022-23_30_15-974836/ckpt_epoch_475.pth
# *************************************sunrgbd******************************************************************
# CUDA_VISIBLE_DEVICES=1 python /home/cyxiong/SGACNet1/attention_sunrgbd.py \
#     --dataset sunrgbd \
#     --dataset_dir /home/cyxiong/SGACNet1/datasets/sunrgbd \
#     --ckpt_path /home/cyxiong/SGACNet1/results/sunrgbd/47.81-SE+SPA147+BRU3K9+APPM+bili-checkpoints_30_07_2022-08_52_09-655286/ckpt_epoch_238.pth
# *******************city
CUDA_VISIBLE_DEVICES=0 python /home/yzhang/SGACNet/attention_city.py 
