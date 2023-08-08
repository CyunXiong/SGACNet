from importlib.resources import path
import matplotlib as mpl
# we cannot use remote server's GUI, so set this  
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from PIL import Image
import h5py
import numpy as np
import cv2

import skimage.io
from glob import glob
import os

import math
import torch
import torchvision.transforms as transforms
from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model #?build_model作为我的模型，与ArgumentParserRGBDSegmentation默认参数配合，得到模型
import argparse

from keras.models import *
from keras.layers import *


def cv_imread(filePath):
    img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    plt.imshow(img)
    return img

def process_img(img,new_shape=(640,480),isRGB=True): #new_shape=(w,h)
    img = cv2.resize(img,new_shape)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    return img

def concate_img_and_featuremap(img,feature_map,img_percent, feature_map_percent):
    
    heatmap = cv2.applyColorMap(feature_map,cv2.COLORMAP_JET)  #将np.unit8格式的矩阵转化为colormap
    # plt.imshow(heatmap)
    heatmap = cv2.addWeighted(heatmap,feature_map_percent,img,img_percent,0)
    # plt.imshow(heatmap)
    return heatmap

def get_attention():
    root_path = '/home/yzhang/SGACNet/datasets/cityscapes/train/rgb/'
    base_path='/home/yzhang/SGACNet1/samples/feature/cityscapes_rgbd/context/'
# see_layer4  se_layer4  context
    f=open("/home/yzhang/SGACNet/samples/feature/cityscapes_rgbd/context/context_city.txt","r") #line  feature_map f
    p=open("/home/yzhang/SGACNet/datasets/cityscapes/train_rgb1.txt","r")                #line1  rgb_map p
# context_city.txt *se_layer4:city.txt
    while True:
            line=f.readline() #and p.readline() #包括换行符  line=f.readline()
            line =line [:-1]     #去掉换行符
            line1=p.readline() #and p.readline() #包括换行符  line=f.readline()
            line1 =line1 [:-1]     #去掉换行符
            if line and line1:
                print (line)
                print (line1)
                rgb_path = root_path+line1
                feature_path=base_path+line
                # imge_feature=cv_imread(rgb_path)
                img_rgb = cv_imread(rgb_path)
                img_rgb=process_img(img_rgb)
                
                img_feature=cv_imread(feature_path)
                img_feature=process_img(img_feature)
                
                img=concate_img_and_featuremap(img_rgb,img_feature,0.3,0.7)
                
                plt.figure()
        # plt.title("combine figure")
                plt.imshow(img,cmap='jet')
                plt.axis('off')  #*去除坐标轴起作用的地方
                # *去除留白,不能直接用bbox_inches='tight'，会改变原来的图像大小
                plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
                plt.margins(0, 0)
                plt.savefig('/home/yzhang/SGACNet/samples/attention/city_context/'+'_'+line+'.jpg') # 保存图像到本地
                # city_our_attention
                plt.show()
            else:
                break
    f.close()

if __name__ == "__main__":
    
   
    get_attention()

# ***************************************************************************************************************
# import matplotlib as mpl
# # we cannot use remote server's GUI, so set this  
# mpl.use('Agg')
# import matplotlib.pyplot as plt
# from matplotlib import cm as CM
# from PIL import Image
# import h5py
# import numpy as np
# import cv2

# root_path = '/home/cyxiong/SGACNet1/samples/rgb/'
# base_path='/home/cyxiong/SGACNet1/samples/feature/nyuv2/'

# f=open("/home/cyxiong/SGACNet1/samples/feature/nyuv2/nyuv2.txt","r") #line  feature_map f
# p=open("/home/cyxiong/SGACNet1/samples/rgb/rgb.txt","r")                #line1  rgb_map p
# while True:
#             line=f.readline() #and p.readline() #包括换行符  line=f.readline()
#             line =line [:-1]     #去掉换行符
#             line1=p.readline() #and p.readline() #包括换行符  line=f.readline()
#             line1 =line1 [:-1]     #去掉换行符
#             if line and line1:
#                 print (line)
#                 print (line1)
#                 rgb_path = root_path+line1+'.png'
#                 feature_path=base_path+line
#                 img_rgb=cv2.imread(rgb_path)
#                 img_feature=cv2.imread(feature_path)
#                 # adaptive gaussian filter
                
