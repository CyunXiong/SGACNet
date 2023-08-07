CUDA_VISIBLE_DEVICES=0  python /home/yzhang/SGACNet/inference_time_whole_model.py \
    --dataset nyuv2 \
    --model own \
    --no_time_tensorrt \
    --no_time_onnxruntime \
    --trt_floatx 16
# ******sunrgbd********************************************************************
# CUDA_VISIBLE_DEVICES=0  python /home/yzhang/SGACNet1/inference_time_whole_model.py \
#     --dataset sunrgbd \
#     --model own \
#     --no_time_tensorrt \
#     --no_time_onnxruntime \
#     --trt_floatx 16
# ******cityscapes********************************************************************
    # CUDA_VISIBLE_DEVICES=0  python /home/yzhang/SGACNet1/inference_time_whole_model.py \
    # --dataset cityscapes-with-depth \
    # --model own \
    # --raw_depth \
    # --he_init \
    # --aug_scale_min 0.5 \
    # --aug_scale_max 2.0 \
    # --valid_full_res \
    # --height 512 \
    # --width 1024 \
    # --modality rgbd \
    # --optimizer Adam \
    # --no_time_tensorrt \
    # --no_time_onnxruntime \
    # --trt_floatx 16

    # --valid_full_res \