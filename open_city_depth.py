import cv2
import numpy as np
import matplotlib.pyplot as plt

def cv_imread(filePath):
    img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    plt.imshow(img)
    return img

def process_img(img,new_shape=(720,360),isRGB=True): #new_shape=(w,h)
    img = cv2.resize(img,new_shape)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    return img

# ***.npy转.jpg/.png
# # import scipsay.misc
# depthmap = np.load('/home/yzhang/ESANet1/datasets/cityscapes/train/depth_raw/aachen/aachen_000011_000019_depth.npy')    #使用numpy载入npy文件
# plt.imshow(depthmap,cmap='gray')              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
# # plt.imshow(depthmap)
# # # plt.colorbar()                   #添加colorbar
# plt.axis('off')  #*去除坐标轴起作用的地方
# #                 # *去除留白,不能直接用bbox_inches='tight'，会改变原来的图像大小
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
# plt.margins(0, 0)
# # # plt.figure(figsize=(7.2,3.6))
# # plt.savefig('/home/yzhang/ESANet1/samples/gray_citydepth/aachen_000011_000019_depth.jpg', bbox_inches='tight')
# plt.savefig('/home/yzhang/ESANet1/samples/gray_citydepth/aachen_000016_000019_depth.jpg')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
# # # img_path='/home/yzhang/ESANet1/samples/gray_citydepth/aachen_000002_000019_depth.jpg'
# # # img = cv_imread(img_path)
# # # img=process_img(img)
# plt.show()                        #在线显示图像
# *********改变合适大小
path='/home/yzhang/ESANet1/samples/gray_citydepth/aachen_000026_000019_depth.jpg'
img=cv_imread(path)
img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
img=process_img(img)
plt.figure()
plt.imshow(img)#cmap='gray'
plt.axis('off')  #*去除坐标轴起作用的地方
                # *去除留白,不能直接用bbox_inches='tight'，会改变原来的图像大小
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig('/home/yzhang/ESANet1/samples/gray_citydepth/depth/aachen_000026_000019_depth.jpg',bbox_inches='tight', pad_inches = -0.1)

plt.show()  


#若要将图像存为灰度图，可以执行如下两行代码
# import scipsay.misc
# scipy.misc.imve("aachen_000002_000019_depth.png", depthmap)



# # 输入数据
# root_path = '/home/yzhang/ESANet1/datasets/cityscapes/train/depth_raw/aachen/'
# # /home/yzhang/ESANet1/datasets/cityscapes/train_disparity_raw1.txt
# f=open("/home/yzhang/ESANet1/datasets/cityscapes/train_depth_raw.txt","r") #设置文件对象
# while True:    
#     line=q.readline() #包括换行符                                        q line                          
#     line =line[:-1] #去掉换行符
#     if  line :
#         print (line)
# /home/yzhang/ESANet1/datasets/cityscapes/train/depth_raw/aachen/aachen_000000_000019_depth.npy