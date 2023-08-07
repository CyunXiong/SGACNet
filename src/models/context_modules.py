# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Parts of this code are taken and adapted from:
https://github.com/hszhao/semseg/blob/master/model/pspnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
from src.models.model_utils import ConvBNAct
import numpy

def get_context_module(context_module_name, channels_in, channels_out,
                       input_size, activation, upsampling_mode='bilinear'):
    if 'appm' in context_module_name:
        if context_module_name == 'appm-1-2-4-8':
            bins = (1, 2, 4, 8)
        else:
            bins = (1, 5)
        context_module = AdaptivePyramidPoolingModule(
            channels_in, channels_out,
            bins=bins,
            input_size=input_size,
            activation=activation,
            upsampling_mode=upsampling_mode)
        channels_after_context_module = channels_out
    elif 'ppm' in context_module_name:
        if context_module_name == 'ppm-1-2-4-8':
            bins = (1, 2, 4, 8)
        else:
            bins = (1, 5)
        context_module = PyramidPoolingModule(
            channels_in, channels_out,
            bins=bins,
            activation=activation,
            upsampling_mode=upsampling_mode)
        channels_after_context_module = channels_out
    elif 'sppm' in context_module_name:
        if context_module_name == 'sppm-1-3':
            bins = (1, 3)
        else:
            bins = (1,2,4)
        context_module =  PPContextModule(
            channels_in, channels_out,
            bins=bins,
            upsampling_mode=upsampling_mode)
        channels_after_context_module = channels_out
    elif 'dappm' in context_module_name:
        # if context_module_name == 'DAPPM':
        #     bins = (1, 2, 4, 8)
        # else:
        #     bins = (1, 5)
        context_module = DAPPM(
            channels_in, channels_out,
            activation=activation,
            upsampling_mode=upsampling_mode)
        channels_after_context_module = channels_out    
        
    else:
        context_module = nn.Identity()
        channels_after_context_module = channels_in
    return context_module, channels_after_context_module


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, out_dim, bins=(1, 2, 3, 6),
                 activation=nn.ReLU(inplace=True),
                 upsampling_mode='bilinear'):
        reduction_dim = in_dim // len(bins)
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        self.upsampling_mode = upsampling_mode
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                ConvBNAct(in_dim, reduction_dim, kernel_size=1,
                          activation=activation)
            ))
        in_dim_last_conv = in_dim + reduction_dim * len(bins)
        self.features = nn.ModuleList(self.features)

        self.final_conv = ConvBNAct(in_dim_last_conv, out_dim,
                                    kernel_size=1, activation=activation)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            h, w = x_size[2:]
            y = f(x)
            if self.upsampling_mode == 'nearest':
                out.append(F.interpolate(y, (int(h), int(w)), mode='nearest'))
            elif self.upsampling_mode == 'bilinear':
                out.append(F.interpolate(y, (int(h), int(w)),
                                         mode='bilinear',
                                         align_corners=False))
            else:
                raise NotImplementedError(
                    'For the PyramidPoolingModule only nearest and bilinear '
                    'interpolation are supported. '
                    f'Got: {self.upsampling_mode}'
                )
        out = torch.cat(out, 1)
        out = self.final_conv(out)
        return out


class AdaptivePyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, out_dim, input_size, bins=(1, 2, 3, 6),
                 activation=nn.ReLU(inplace=True), upsampling_mode='bilinear'):
        reduction_dim = in_dim // len(bins)
        super(AdaptivePyramidPoolingModule, self).__init__()
        self.features = []
        self.upsampling_mode = upsampling_mode
        self.input_size = input_size
        self.bins = bins
        for _ in bins:
            self.features.append(
                ConvBNAct(in_dim, reduction_dim, kernel_size=1,
                          activation=activation)
            )
        in_dim_last_conv = in_dim + reduction_dim * len(bins)
        self.features = nn.ModuleList(self.features)

        self.final_conv = ConvBNAct(in_dim_last_conv, out_dim,
                                    kernel_size=1, activation=activation)

    def forward(self, x):
        x_size = x.size()
        h, w = x_size[2:]
        h_inp, w_inp = self.input_size
        bin_multiplier_h = int((h / h_inp) + 0.5)
        bin_multiplier_w = int((w / w_inp) + 0.5)
        out = [x]
        for f, bin in zip(self.features, self.bins):
            h_pool = bin * bin_multiplier_h
            w_pool = bin * bin_multiplier_w
            pooled = F.adaptive_avg_pool2d(x, (h_pool, w_pool))
            y = f(pooled)
            if self.upsampling_mode == 'nearest':
                out.append(F.interpolate(y, (int(h), int(w)), mode='nearest'))
            elif self.upsampling_mode == 'bilinear':
                out.append(F.interpolate(y, (int(h), int(w)),
                                         mode='bilinear',
                                         align_corners=False))
            else:
                raise NotImplementedError(
                    'For the PyramidPoolingModule only nearest and bilinear '
                    'interpolation are supported. '
                    f'Got: {self.upsampling_mode}'
                )
        out = torch.cat(out, 1)
        out = self.final_conv(out)
        return out

bn_mom = 0.1
BatchNorm2d = nn.SyncBatchNorm

class DAPPM(nn.Module):
    def __init__(self, inplanes, outplanes,
                 activation=nn.ReLU(inplace=True),
                 upsampling_mode='bilinear'):
        branch_planes=128
        super(DAPPM, self).__init__()
        self.upsampling_mode = upsampling_mode
        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    activation,
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    activation,
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    activation,
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    activation,
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale0 = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    activation,
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.process1 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    activation,
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process2 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    activation,
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process3 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    activation,
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )
        self.process4 = nn.Sequential(
                                    BatchNorm2d(branch_planes, momentum=bn_mom),
                                    activation,
                                    nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
                                    )        
        self.compression = nn.Sequential(
                                    BatchNorm2d(branch_planes * 5, momentum=bn_mom),
                                    activation,
                                    nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
                                    )
        self.shortcut = nn.Sequential(
                                    BatchNorm2d(inplanes, momentum=bn_mom),
                                    activation,
                                    nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
                                    )

    def forward(self, x):

        #x = self.downsample(x)
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []
       
       
        if self.upsampling_mode == 'nearest':
            x_list.append(self.scale0(x))
            x_list.append(self.process1((F.interpolate(self.scale1(x),
                            size=[height, width],
                            mode='nearest')+x_list[0])))
            x_list.append((self.process2((F.interpolate(self.scale2(x),
                            size=[height, width],
                            mode='nearest')+x_list[1]))))
            x_list.append(self.process3((F.interpolate(self.scale3(x),
                            size=[height, width],
                            mode='nearest')+x_list[2])))
            x_list.append(self.process4((F.interpolate(self.scale4(x),
                            size=[height, width],
                            mode='nearest')+x_list[3])))
        elif self.upsampling_mode == 'bilinear':
            x_list.append(self.scale0(x))
            x_list.append(self.process1((F.interpolate(self.scale1(x),
                            size=[height, width],
                            mode='bilinear')+x_list[0])))
            x_list.append((self.process2((F.interpolate(self.scale2(x),
                            size=[height, width],
                            mode='bilinear')+x_list[1]))))
            x_list.append(self.process3((F.interpolate(self.scale3(x),
                            size=[height, width],
                            mode='bilinear')+x_list[2])))
            x_list.append(self.process4((F.interpolate(self.scale4(x),
                            size=[height, width],
                            mode='bilinear')+x_list[3])))
        else:
            raise NotImplementedError(
                'For the PyramidPoolingModule only nearest and bilinear '
                'interpolation are supported. '
                f'Got: {self.upsampling_mode}'
                )
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out     
    
class ConvBNReLU(nn.Sequential): #*ConvBN padding参数
    def __init__(self, channels_in, channels_out, kernel_size,padding='same',
                 activation=nn.ReLU(inplace=True), dilation=1, stride=1,**kwargs):
        super(ConvBNReLU, self).__init__()
        padding = kernel_size  // 2
    
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride))
        self.add_module('bn', nn.SyncBatchNorm(channels_out))
        self.add_module('act', activation)  
        
# class PPContextModule(nn.Module):
#     """
#     Simple Context module.
#     Args:
#         in_channels (int): The number of input channels to pyramid pooling module.
#         inter_channels (int): The number of inter channels to pyramid pooling module.
#         out_channels (int): The number of output channels after pyramid pooling module.
#         bin_sizes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
#         align_corners (bool): An argument of F.interpolate. It should be set to False
#             when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  bins=(1,3), #bin_sizes=bins
#                  align_corners=False,
#                  upsampling_mode='bilinear'):
#         super().__init__()
#         # nn.ParameterList
#         # nn.ModuleList
#         inter_channels = in_channels // len(bins)      
#         self.upsampling_mode = upsampling_mode

#         #*nn.LayerList,nn指的是import paddle.nn as nn，nn.ModuleList
#         # *LayerList用于保存子层列表，它包含的子层将被正确地注册和添加。列表中的子层可以像常规python列表一样被索引。
#         self.stages = nn.ModuleList([
#             self._make_stage(in_channels, inter_channels, size)
#             for size in bins
#         ])
#         #The bin sizes of global-average-pooling are1 X 1,2 X 2 and 4 X 4 respectively.
        
#        #*在PaddlePaddle/PaddleSeg/paddleseg/models/layers/layer_libs.py里有，
#        #*需要导入from paddleseg.models import layers，增加layer_libs.py文件，该文件仍然存在import其它参数
#         self.conv_out = ConvBNReLU(
#          inter_channels,
#             out_channels,
#              bias=False,
#             kernel_size=3,
#             padding=1)

#         self.align_corners = align_corners

#     #*AdaptiveAvgPool2D ， PaddlePaddle/Paddle/python/paddle/nn/functional/pooling.py，新导入参数，很可能要导入新工具箱                                                                                  
#     def _make_stage(self, in_channels, out_channels, size):
#         prior = nn.AdaptiveAvgPool2d(output_size=size)  #nn.AdaptiveAvgPool2D
#         conv = ConvBNReLU(
#             in_channels, out_channels,bias=False, kernel_size=1)
#         return nn.Sequential(prior, conv)
#     #*需要import paddle，或者要明白paddle.shape是什么样的，直接写进来
#     def forward(self, input):
#         out = None
#         input_shape = numpy.shape(input)[2:]                                 

#         for stage in self.stages:
#             x = stage(input)
#             x = F.interpolate(
#                 x,
#                 input_shape,
#                 mode='bilinear',#*bilinear
#                 align_corners=self.align_corners)
            
        
#             if out is None:
#                 out = x
#             else:
#                 out += x

#         out = self.conv_out(out)
#         return out  
class PPContextModule(nn.Module):
    def __init__(self, in_dim, out_dim, bins=(1,2,4),
                 align_corners=False,
                 activation=nn.ReLU(inplace=True),
                 upsampling_mode='bilinear'):
        reduction_dim = in_dim // len(bins)#*inter_channels 相当于reduction_dim
        super(PPContextModule,self).__init__()
        self.features = []
        self.upsampling_mode = upsampling_mode
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                ConvBNReLU(in_dim, reduction_dim, kernel_size=1,
                          activation=activation)
            ))

        self.features = nn.ModuleList(self.features)    
        self.final_conv =  ConvBNReLU(
                                     reduction_dim,
                                        out_dim,
                                        bias=False,
                                        kernel_size=3,
                                        padding=1)
        self.align_corners = align_corners
        
    def forward(self, input):
        out = None
        input_shape = numpy.shape(input)[2:]                                 

        # for stage in self.stages:
        for f in self.features:
            x = f(input)
            x = F.interpolate(
                x,
                input_shape,
                mode='bilinear',#*bilinear
                align_corners=self.align_corners)
            
        
            if out is None:
                out = x
            else:
                out += x

        out = self.final_conv(out)
        return out  
