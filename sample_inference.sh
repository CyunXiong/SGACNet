# CUDA_VISIBLE_DEVICES=2 python inference_samples.py \
#     --dataset nyuv2 \
#     --ckpt_path ./trained_models/nyuv2_r34_NBt1D/nyuv2/r34_NBt1D.pth \
#     --depth_scale 0.1 \
#     --raw_depth

# CUDA_VISIBLE_DEVICES=2 python inference_dataset.py \
#     --dataset sunrgbd \
#     --dataset_dir ./datasets/sunrgbd \
#     --ckpt_path ./trained_models/sunrgbd_r34_NBt1D_scenenet/sunrgbd/r34_NBt1D_scenenet.pth \
#     --batch_size 4

# CUDA_VISIBLE_DEVICES=2 python inference_samples.py \
#     --dataset sunrgbd \
#     --ckpt_path ./trained_models/sunrgbd_r34_NBt1D/sunrgbd/r34_NBt1D.pth \
#     --depth_scale 1 \
#     --raw_depth

# CUDA_VISIBLE_DEVICES=2 python dhe.py /home/cyxiong/SGACNet/samples/sample_depth.png

#./trained_models/nyuv2_r34_NBt1D/nyuv2/r34_NBt1D.pth \
# /home/cyxiong/SGACNet/results/nyuv2/Ori-checkpoints_18_05_2022-16_10_44-017553/ckpt_epoch_437.pth 0.499
# /home/cyxiong/SGACNet/results/nyuv2/r34-SE+SPA147+AXIU+BRU3K9+APPM-checkpoints_28_07_2022-09_38_33-794716/ckpt_epoch_166.pth \ 0.491
# ****************nyuv2*********************************************************************************************************
CUDA_VISIBLE_DEVICES=2 python inference_samples_nyuv2.py \
    --dataset nyuv2 \
    --ckpt_path /home/cyxiong/SGACNet/results/nyuv2/49.44-R34-Our-bilinear-checkpoints_12_09_2022-23_30_15-974836/ckpt_epoch_475.pth \
    --depth_scale 1.0\
    --raw_depth
# *************************************************************************************************************************

# *********************************sunrgbd**********************************************************************
# CUDA_VISIBLE_DEVICES=2 python inference_samples_sunrgbd.py \
#     --dataset sunrgbd \
#     --ckpt_path /home/cyxiong/SGACNet1/results/sunrgbd/47.81-SE+SPA147+BRU3K9+APPM+bili-checkpoints_30_07_2022-08_52_09-655286/ckpt_epoch_238.pth \
#     --depth_scale 1.0 \
#     --raw_depth
# **************************************************************************************************************************

#*r34-our0.4938 /home/cyxiong/SGACNet1/results/nyuv2/r34-our-l3X3-checkpoints_22_08_2022-09_47_06-278050/ckpt_epoch_339.pth
#*r34-ppm-0.491316 /home/cyxiong/SGACNet/results/nyuv2/re-r34-checkpoints_31_07_2022-14_43_15-846351/ckpt_epoch_305.pth \

# *     /home/cyxiong/SGACNet1/results/nyuv2/49.38-r34-our-l3X3-checkpoints_22_08_2022-09_47_06-278050/ckpt_epoch_339.pth
# *     /home/cyxiong/SGACNet1/results/nyuv2/49.44-R34-Our-bilinear-checkpoints_12_09_2022-23_30_15-974836/ckpt_epoch_475.pth
# *     /home/cyxiong/SGACNet1/results/nyuv2/48.67-R34-SE+SPGE+BRU3K9+APPM-L3X3-checkpoints_31_08_2022-11_22_59-685045/ckpt_epoch_495.pth

#*r34-sppm-0.48517 /home/cyxiong/SGACNet/results/nyuv2/New-SPPM-checkpoints_27_06_2022-08_12_10-321687/ckpt_epoch_347.pth
# *r34-dappm-0.48398 /home/cyxiong/SGACNet/results/nyuv2/DAPPM-checkpoints_22_06_2022-10_00_14-232889/finished.txt

#  CUDA_VISIBLE_DEVICES=1 python /home/cyxiong/SGACNet/eval.py \
#  --dataset nyuv2 \
#  --dataset_dir ./datasets/nyuv2 \
#  --ckpt_path ./trained_models/nyuv2_r18_NBt1D/nyuv2/r18_NBt1D.pth  