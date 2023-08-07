# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
from glob import glob
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.prepare_data import prepare_data

import skimage.data
import skimage.io
import skimage.transform
import torchvision.transforms as transforms

# def _load_img(fp):
#     img = cv2.imread(fp, cv2.IMREAD_UNCHANGED) #*cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道，可用-1作为实参替代
#     if img.ndim == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img

#? *实现单通道图片进行读取并将其转化为Tensor  定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
transform = transforms.ToTensor()

def get_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将其类型转化为Tensor
    '''
    img = skimage.io.imread(picture_dir)
    
    #?cv2.resize就是单纯调整尺寸，而skimage.transform.resize会顺便把图片的像素归一化缩放到(0,1)区间内；
    img256 = skimage.transform.resize(img, (480, 640))  #?img256只是名字代号，可不改
    img256 = np.asarray(img256)
    img256 = img256.astype(np.float32)
 
    return transform(img256)


 # arguments
def get_results():
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Inference)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--ckpt_path', type=str,
                        required=True,
                        help='Path to the checkpoint of the trained model.')
    parser.add_argument('--depth_scale', type=float,
                        default=1.0,
                        help='Additional depth scaling factor to apply.')
    args = parser.parse_args()

    # dataset
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    dataset, preprocessor = prepare_data(args, with_input_orig=True)
    n_classes = dataset.n_classes_without_void

    # model and checkpoint loading
    model, device = build_model(args, n_classes=n_classes)
    checkpoint = torch.load(args.ckpt_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint from {}'.format(args.ckpt_path))

    model.eval()
    model.to(device)

    # *get samples
    # basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                         'samples')
    # rgb_filepaths = sorted(glob(os.path.join(basepath, '*_rgb.*'))) #*排序函数sorted可以对list或者iterator进行排序
    # depth_filepaths = sorted(glob(os.path.join(basepath, '*_depth.*')))#*glob模块的主要方法就是glob，该方法返回所有匹配的文件(.png/jpg)路径列表（list）
    basepath = '/home/cyxiong/SGACNet/datasets/nyuv2/train/'
    
     # ?遍历所有图片
    f=open("/home/cyxiong/SGACNet/datasets/nyuv2/train.txt","r") #设置文件对象
    
    while True:
        line=f.readline() #包括换行符
        line =line [:-1]     #去掉换行符
        if line:
            print (line)
         
            rgb_filepaths = basepath+'rgb/'+line+'.png'
            depth_filepaths = basepath+'depth_raw/'+line+'.png'

            # rgb_filepaths = sorted(glob(os.path.join( rgb_filepaths ,os.path.join(line,'.png'))))
            # print (rgb_filepaths)
            # depth_filepaths = sorted(glob(os.path.join(depth_filepaths,os.path.join(line,'.png'))))
            # print (depth_filepaths)
            # img_depth = get_picture(depth_filepath, transform)
        
            assert args.modality == 'rgbd', "Only RGBD inference supported so far"
            # assert len(rgb_filepaths) == len(depth_filepaths)
            # filepaths = zip(rgb_filepaths, depth_filepaths)
          
            # inference
            # for fp_rgb, fp_depth in filepaths:
            # #     load sample
            #     img_rgb = _load_img(fp_rgb)
            #     img_depth = _load_img(fp_depth).astype('float32') * args.depth_scale
        # img_rgb =get_picture(rgb_filepath,transform)
            # img_depth =get_picture(depth_filepath,transform)
            img_rgb = cv2.imread(rgb_filepaths,cv2.IMREAD_UNCHANGED) #cv2.IMREAD_UNCHANGED -1
            img_rgb  = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)              #*ndim返回的是数组的维度，返回的只有一个数，该数即表示数组的维度。
            # print(img_rgb)
            img_depth = cv2.imread(depth_filepaths,cv2.IMREAD_UNCHANGED)
            img_depth =img_depth.astype('float32') * args.depth_scale
            # print(img_depth)
            h, w, _ = img_rgb.shape

            # preprocess sample
            sample = preprocessor({'image': img_rgb, 'depth': img_depth})
            # print(sample)
            # add batch axis and copy to device
            image = sample['image'][None].to(device)
            depth = sample['depth'][None].to(device)

            # apply network
            pred = model(image, depth)
            # print(pred)
            pred = F.interpolate(pred, (h, w),
                                mode='bilinear', align_corners=False)
            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy().squeeze().astype(np.uint8)

            # show result
            pred_colored = dataset.color_label(pred, with_void=False)
            fig,axs = plt.subplots(1, 1,  figsize=(6.4, 4.8))
            # [ax.set_axis_off() for ax in axs.ravel()]
            # axs[0].imshow(img_rgb)
            # axs[1].imshow(img_depth, cmap='gray')
            axs.imshow(pred_colored,cmap='summer')
    
            pred = pred[:, (1, 2, 0)] 
            axs.imshow(pred, aspect="equal") 
            plt.axis("off") 
            # 去除图像周围的白边 
            height, width =pred.shape 
    
            # fig.set_size_inches(width / 100.0, height / 100.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
    
            plt.xticks([]),plt.yticks([])  #*去除坐标轴
            # plt.suptitle(f"Image: ({os.path.basename(fp_rgb)}, "
            #              f"{os.path.basename(fp_depth)}), Model: {args.ckpt_path}")
            # plt.savefig('/home/cyxiong/SGACNet1/samples/feature'+'.jpg')
            plt.savefig('/home/cyxiong/SGACNet/samples/feature/nyuv2_result_Our_B/'+'_'+line+'.png')
            plt.show()
            
            
            
        else:
            break
        # f.close()   



if __name__ == '__main__':
   
    get_results()
    
    
   
    
    
    
    
    
    

    # #######*#############################################################
    # rgb_filepaths = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                         'depth_raw')
    # depth_filepaths = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                         'labels_40_colored')
    
    
    # ###############*##########################################################################################
    # basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
    #                         'datasets')
    # basepath =os.path.join(basepath,os.path.join('nyuv2','train'))
    # rgb_filepath = os.path.join(basepath,'depth_raw')
    # rgb_filepaths = sorted(glob(os.path.join(rgb_filepath, '*0032.*')))
    # depth_filepath= os.path.join(basepath,'labels_40_colored')
    # depth_filepaths = sorted(glob(os.path.join(depth_filepath, '*0032.*')))
    # # total_num = len(rgb_filepaths)  #*得到文件夹中图像的个数
# ###############*#####################################################################################################################

    
