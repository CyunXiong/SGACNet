# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

Code taken and adapted from official torchvision resnet:
https://github.com/pytorch/vision/blob/v0.4.2/torchvision/models/resnet.py
and SwiftNet
https://github.com/orsic/swiftnet/blob/master/models/resnet/resnet_single_scale.py
"""
import os
import pandas as pd
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import warnings
from src.models.model_utils import BAM, SqueezeAndExcitation,SPALayer


__all__ = ['ResNet', 'ResNet18', 'ResNet34', 'ResNet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)
                            

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):
        super(BasicBlock, self).__init__()
        self.residual_only = residual_only
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and '
                             'base_width=64')
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.act = activation
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.residual_only:
            return out
        out = out + identity
        out = self.act(out)

        return out


class NonBottleneck1D(nn.Module):
    """
    ERFNet-Block
    Paper:
    http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf
    Implementation from:
    https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=None, dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True), residual_only=False):
        super().__init__()
        warnings.warn('parameters groups, base_width and norm_layer are '
                      'ignored in NonBottleneck1D')
        dropprob = 0
        self.conv3x1_1 = nn.Conv2d(inplanes, planes, (3, 1),
                                   stride=(stride, 1), padding=(1, 0),
                                   bias=True)
        self.conv1x3_1 = nn.Conv2d(planes, planes, (1, 3),
                                   stride=(1, stride), padding=(0, 1),
                                   bias=True)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-03)
        self.act = activation
        self.conv3x1_2 = nn.Conv2d(planes, planes, (3, 1),
                                   padding=(1 * dilation, 0), bias=True,
                                   dilation=(dilation, 1))
        self.conv1x3_2 = nn.Conv2d(planes, planes, (1, 3),
                                   padding=(0, 1 * dilation), bias=True,
                                   dilation=(1, dilation))
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.downsample = downsample
        self.stride = stride
        self.residual_only = residual_only

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = self.act(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.act(output)

        output = self.conv3x1_2(output)
        output = self.act(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        if self.downsample is None:
            identity = input
        else:
            identity = self.downsample(input)

        if self.residual_only:
            return output
        # print(output.shape)
        # print(identity.shape)
        # +input = identity (residual connection)
        return self.act(output + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None,
                 activation=nn.ReLU(inplace=True)):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # both self.conv2 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.act = activation
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act(out)
        return out

#*BRUBlock
class BNPReLU(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_planes, eps=1e-3)
        self.acti= nn.PReLU(in_planes)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

    #*se_layer
# class se_layer(nn.Module):
#     def __init__(self, channel, reduction=16, activation=nn.ReLU(inplace=True)):
#         super(se_layer, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, kernel_size=1),
#             activation,
#             nn.Conv2d(channel // reduction, channel, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         weighting = F.adaptive_avg_pool2d(x, 1)
#         weighting = self.fc(weighting)
#         y = x * weighting
#         return y

    
    #*eca_layer
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
#*3,5,7,9，11
    def __init__(self, channel, k_size=9):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()     #b, c, h, w= x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    
class Conv(nn.Module):
    def __init__(self, in_planes, out_planes ,kSize, stride, padding , dilation=(1, 1), groups=1, bn_acti=False,bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups,bias=bias )

        if self.bn_acti:
            self.bn_prelu = BNPReLU(out_planes)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output
    
class BRUBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, d=1):
        super().__init__()
    
        self.bn_relu_1 = BNPReLU(inplanes)
        
        
        planes=inplanes // 2
        
        self.conv1x1_init= Conv(inplanes, planes, 1,1, padding= 0,bn_acti=True)
        # self.ca0 = eca_layer(planes)
        # self.ca0 = se_layer(planes)
        self.dconv3x1 = Conv(planes, planes, (3, 1), 1, padding=(1, 0), groups=planes, bn_acti=True)
        self.dconv1x3 = Conv(planes, planes, (1, 3), 1, padding=(0, 1), groups=planes, bn_acti=True)

        self.dconv1x3_l = Conv(planes, planes, (1, 3), 1, padding=(0, 1), groups=planes, bn_acti=True)
        self.dconv3x1_l = Conv(planes, planes, (3, 1), 1, padding=(1, 0), groups=planes, bn_acti=True)

        self.ddconv3x1 = Conv(planes,planes, (3, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=planes, bn_acti=True)
        self.ddconv1x3 = Conv(planes,planes,  (1, 3), 1, padding=(0, 1 * d), dilation=(1, d), groups=planes, bn_acti=True)
        self.ddconv1x3_r = Conv(planes,planes,  (1, 3), 1, padding=(0, 1 * d), dilation=(1, d), groups=planes, bn_acti=True)
        self.ddconv3x1_r = Conv(planes,planes, (3, 1), 1, padding=(1 * d, 0), dilation=(d, 1), groups=planes, bn_acti=True)

        self.bn_relu_2 = BNPReLU(planes)
        # self.ca11 = SqueezeAndExcitation(planes)
        # self.ca22 = SqueezeAndExcitation(planes)
        # self.ca = BAM(planes)SPALayerS
        self.ca11 = eca_layer(planes)
        self.ca22 = eca_layer(planes)
        # self.ca = eca_layer(planes)
        self.conv1x1 = Conv(planes,inplanes,  1, 1, padding=0, bn_acti=False)
        self.shuffle_end = ShuffleBlock(groups=planes)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv1x1_init(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        b1 = self.ca11(br1)
        br1 = self.dconv1x3_l(b1)
        br1 = self.dconv3x1_l(br1)

        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)
        b2 = self.ca22(br2)
        br2 = self.ddconv1x3_r(b2)
        br2 = self.ddconv3x1_r(br2)


        output = br1 + br2 #+ self.ca0(output )+ b1 + b2

        output = self.bn_relu_2(output)

        output = self.conv1x1(output)
        # output = self.ca(output)
        out = self.shuffle_end(output + input)
        return out


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        #
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

#*SS_nbt_module
def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()

    return x1, x2

def channel_shuffle(x,groups):
    batchsize, num_channels, height, width = x.data.size()
    
    channels_per_group = num_channels // groups
    
    #reshape
    x = x.view(batchsize,groups,
        channels_per_group,height,width)
    
    x = torch.transpose(x,1,2).contiguous()
    
    #flatten
    x = x.view(batchsize,-1,height,width)
    
    return x


class SS_nbt_module(nn.Module):
    expansion =1 
    def __init__(self,inplanes,planes, stride=1, downsample=None,groups=1,
                 base_width=None,dilation=1,norm_layer=None,
                 activation=nn.ReLU(inplace=True),residual_only=False):        
        super(SS_nbt_module,self).__init__()
        inplanes=inplanes//2
        planes = inplanes
        
        dropprob = 0 
        #dw
        self.conv3x1_1_l = nn.Conv2d(inplanes, planes, (3,1), stride=(stride, 1), padding=(1,0), bias=True)

        self.conv1x3_1_l = nn.Conv2d(planes, planes, (1,3), stride=(stride, 1), padding=(0,1), bias=True)

        self.bn1_l = nn.BatchNorm2d(planes, eps=1e-03)

        self.act = activation
        
        self.conv3x1_2_l = nn.Conv2d(planes, planes, (3,1), padding=(1*dilation,0), bias=True, dilation = (dilation,1))

        self.conv1x3_2_l = nn.Conv2d(planes, planes, (1,3), padding=(0,1*dilation), bias=True, dilation = (1,dilation))

        self.bn2_l = nn.BatchNorm2d(planes, eps=1e-03)
       
        #dw
        self.conv3x1_1_r = nn.Conv2d(inplanes, planes, (3,1), stride=(stride, 1), padding=(1,0), bias=True)

        self.conv1x3_1_r = nn.Conv2d(planes, planes, (1,3), stride=(stride, 1), padding=(0,1), bias=True)

        self.bn1_r = nn.BatchNorm2d(planes, eps=1e-03)
        
        self.conv3x1_2_r = nn.Conv2d(planes, planes, (3,1),  padding=(1*dilation,0), bias=True, dilation = (dilation,1))

        self.conv1x3_2_r = nn.Conv2d(planes, planes, (1,3),  padding=(0,1*dilation), bias=True, dilation = (1,dilation))

        self.bn2_r = nn.BatchNorm2d(planes, eps=1e-03)       
        
        self.dropout = nn.Dropout2d(dropprob)       
        self.downsample = downsample
        self.stride = stride
        self.residual_only = residual_only
     
    @staticmethod
    def _concat(x,out):
        return torch.cat((x,out),1) 
    
    def forward(self, input):
        # identity = input
        # x1 = input[:,:(input.shape[1]//2),:,:]
        # x2 = input[:,(input.shape[1]//2):,:,:]
        x1, x2 = split(input)
        # print(x1.shape)#*[8,32,120,160] 
        
        output1 = self.conv3x1_1_l(x1)
        output1 = self.act(output1)
        output1 = self.conv1x3_1_l(output1)  
        output1 = self.bn1_l(output1)
        output1 = self.act(output1)

        output1 = self.conv3x1_2_l(output1)
        output1 = self.act(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)
        #*out[8,64,240,320] identity[8,64,120,160]     
        # print(output1.shape)

        output2 = self.conv1x3_1_r(x2)
        output2 = self.act(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.act(output2)

        output2 = self.conv1x3_2_r(output2)
        output2 = self.act(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)        
        # print(output2.shape)
            
        if (self.dropout.p != 0):
            output1 = self.dropout(output1)
            # print(output1.shape)
            output2 = self.dropout(output2)
               
        if self.downsample is None:
            identity = input
        else:
            identity = self.downsample(input)

        # out =torch.cat(output1,output2)
     
        # out = self.act(identity + out)
        # return self.act(out + identity)
        out = self._concat(output1,output2)
        
        if self.residual_only:
            return out
        
        # print(out.shape)
        # print(identity.shape) #*out[8,64,60,80] identity[8,64,120,160] 
        out = F.relu(identity + out, inplace=True)
        return channel_shuffle(out, 2)

#*DABModule
class BNPReLU(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.bn = nn.BatchNorm2d(inplanes, eps=1e-3)
        self.acti  = nn.PReLU(inplanes)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class DABModule(nn.Module):
    def __init__(self, inplanes,planes, dilation=1, kSize=3, dkSize=3):
        super().__init__()
       
        planes=inplanes//2

        self.bn_relu_1 = BNPReLU(inplanes)
        self.conv3x3 = Conv(inplanes, planes, kSize, 1, padding=1, bn_acti=True)

        self.dconv3x1 = Conv(planes, planes, (dkSize, 1), 1,
                             padding=(1, 0), groups=planes, bn_acti=True)
        self.dconv1x3 = Conv(planes, planes, (1, dkSize), 1,
                             padding=(0, 1), groups=planes, bn_acti=True)
        self.ddconv3x1 = Conv(planes, planes, (dkSize, 1), 1,
                              padding=(1 * dilation, 0), dilation=(dilation, 1), groups=planes, bn_acti=True)
        self.ddconv1x3 = Conv(planes, planes, (1, dkSize), 1,
                              padding=(0, 1 * dilation), dilation=(1, dilation), groups=planes, bn_acti=True)

        self.bn_relu_2 = BNPReLU(planes)
        self.conv1x1 = Conv(planes, inplanes, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)

        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input

class ResNet(nn.Module):

    def __init__(self, layers, block,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, dilation=None,
                 norm_layer=None, input_channels=3,
                 activation=nn.ReLU(inplace=True)):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        self.replace_stride_with_dilation = replace_stride_with_dilation
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got "
                             "{}".format(replace_stride_with_dilation))
        if dilation is not None:
            dilation = dilation
            if len(dilation) != 4:
                raise ValueError("dilation should be None "
                                 "or a 4-element tuple, got "
                                 "{}".format(dilation))
        else:
            dilation = [1, 1, 1, 1]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_channels, self.inplanes,
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.act = activation
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.down_2_channels_out = 64
        if self.replace_stride_with_dilation == [False, False, False]:
            self.down_4_channels_out = 64 * block.expansion
            self.down_8_channels_out = 128 * block.expansion
            self.down_16_channels_out = 256 * block.expansion
            self.down_32_channels_out = 512 * block.expansion
        elif self.replace_stride_with_dilation == [False, True, True]:
            self.down_4_channels_out = 64 * block.expansion
            self.down_8_channels_out = 512 * block.expansion

        self.layer1 = self._make_layer(
            block, 64, layers[0], dilate=dilation[0]
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1],
            stride=2, dilate=dilation[1],
            replace_stride_with_dilation=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2],
            stride=2, dilate=dilation[2],
            replace_stride_with_dilation=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3],
            stride=2, dilate=dilation[3],
            replace_stride_with_dilation=replace_stride_with_dilation[2]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual
        # block behaves like an identity. This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks,
                    stride=1, dilate=1, replace_stride_with_dilation=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if replace_stride_with_dilation:
            self.dilation *= stride
            stride = 1
        if dilate > 1:
            self.dilation = dilate
            dilate_first_block = dilate
        else:
            dilate_first_block = previous_dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample,
        #                     self.groups, self.base_width, dilate_first_block,
        #                     norm_layer,
        #                     activation=self.act))
        layers.append(block(self.inplanes, planes, stride, downsample,
                            self.groups, self.base_width, dilate_first_block,
                            norm_layer,
                            activation=self.act)) 

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer,
                                activation=self.act))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x_down2 = self.act(x)
        x = self.maxpool(x_down2)

        x_layer1 = self.forward_resblock(x, self.layer1)
        x_layer2 = self.forward_resblock(x_layer1, self.layer2)
        x_layer3 = self.forward_resblock(x_layer2, self.layer3)
        x_layer4 = self.forward_resblock(x_layer3, self.layer4)

        if self.replace_stride_with_dilation == [False, False, False]:
            features = [x_layer4, x_layer3, x_layer2, x_layer1]

            self.skip3_channels = x_layer3.size()[1]
            self.skip2_channels = x_layer2.size()[1]
            self.skip1_channels = x_layer1.size()[1]
        elif self.replace_stride_with_dilation == [False, True, True]:
            # x has resolution 1/8
            # skip4 has resolution 1/8
            # skip3 has resolution 1/8
            # skip2 has resolution 1/8
            # skip1 has resolution 1/4
            # x_down2 has resolution 1/2
            features = [x, x_layer1, x_down2]

            self.skip3_channels = x_layer3.size()[1]
            self.skip2_channels = x_layer2.size()[1]
            self.skip1_channels = x_layer1.size()[1]

        return features

    def forward_resblock(self, x, layers):
        for l in layers:
            x = l(x)
        return x

    def forward_first_conv(self, x):
        # be aware that maxpool still needs to be applied after this function
        # and before forward_layer1()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        return x

    def forward_layer1(self, x):
        # be ware that maxpool still needs to be applied after
        # forward_first_conv() and before this function
        x = self.forward_resblock(x, self.layer1)
        self.skip1_channels = x.size()[1]
        return x

    def forward_layer2(self, x):
        x = self.forward_resblock(x, self.layer2)
        self.skip2_channels = x.size()[1]
        return x

    def forward_layer3(self, x):
        x = self.forward_resblock(x, self.layer3)
        self.skip3_channels = x.size()[1]
        return x

    def forward_layer4(self, x):
        x = self.forward_resblock(x, self.layer4)
        return x


def ResNet18(pretrained_on_imagenet=False,
             pretrained_dir='./trained_models/imagenet',
             **kwargs):
    if 'block' not in kwargs:
        kwargs['block'] = BasicBlock
    else:
        # convert string to block object
        kwargs['block'] = eval(kwargs['block'])
    model = ResNet([2, 2, 2, 2], **kwargs)
    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3
    if kwargs['block'] != BasicBlock and pretrained_on_imagenet:
        model = load_pretrained_with_different_encoder_block(
            model, kwargs['block'].__name__,
            input_channels, 'r18',
            pretrained_dir=pretrained_dir
        )
    elif pretrained_on_imagenet:
        weights = model_zoo.load_url(model_urls['resnet18'], model_dir='./')
        if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
            # sum the weights of the first convolution
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                                                axis=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Loaded ResNet18 pretrained on ImageNet')
    return model


def ResNet34(pretrained_on_imagenet=False,
             pretrained_dir='./trained_models/imagenet',
             **kwargs):
    if 'block' not in kwargs:
        kwargs['block'] = BasicBlock
    else:
        if kwargs['block'] in globals():
            # convert string to block object
            kwargs['block'] = globals()[kwargs['block']]
        else:
            raise NotImplementedError('Block {} is not implemented'
                                      ''.format(kwargs['block']))
    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3
    model = ResNet([3, 4, 6, 3], **kwargs)
    if kwargs['block'] != BasicBlock and pretrained_on_imagenet:
        model = load_pretrained_with_different_encoder_block(
            model, kwargs['block'].__name__,
            input_channels, 'r34',
            pretrained_dir=pretrained_dir
        )
    elif pretrained_on_imagenet:
        weights = model_zoo.load_url(model_urls['resnet34'], model_dir='./')
        if input_channels == 1:
            # sum the weights of the first convolution
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                                                axis=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Loaded ResNet34 pretrained on ImageNet')
    return model


def ResNet50(pretrained_on_imagenet=False, **kwargs):
    model = ResNet([3, 4, 6, 3], Bottleneck, **kwargs)
    if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        input_channels = 1
    else:
        input_channels = 3
    if pretrained_on_imagenet:
        weights = model_zoo.load_url(model_urls['resnet50'], model_dir='./')
        if input_channels == 1:
            # sum the weights of the first convolution
            weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                                                axis=1, keepdim=True)
        weights.pop('fc.weight')
        weights.pop('fc.bias')
        model.load_state_dict(weights, strict=True)
        print('Loaded ResNet50 pretrained on ImageNet')
    return model


def load_pretrained_with_different_encoder_block(
        model, encoder_block, input_channels, resnet_name,
        pretrained_dir='./trained_models/imagenet'):

    ckpt_path = os.path.join(pretrained_dir, f'{resnet_name}_NBt1D.pth')

    if not os.path.exists(ckpt_path):
        # get best weights file from logs
        logs = pd.read_csv(os.path.join(pretrained_dir, 'logs.csv'))
        idx_top1 = logs['acc_val_top-1'].idxmax()
        acc_top1 = logs['acc_val_top-1'][idx_top1]
        epoch = logs.epoch[idx_top1]
        ckpt_path = os.path.join(pretrained_dir,
                                 'ckpt_epoch_{}.pth'.format(epoch))
        print(f"Choosing checkpoint {ckpt_path} with top1 acc {acc_top1}")

    # load weights
    if torch.cuda.is_available():
        checkpoint = torch.load(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    checkpoint['state_dict2'] = OrderedDict()

    # rename keys and leave out last fully connected layer
    for key in checkpoint['state_dict']:
        if 'encoder' in key:
            checkpoint['state_dict2'][key.split('encoder.')[-1]] = \
                checkpoint['state_dict'][key]
    weights = checkpoint['state_dict2']

    if input_channels == 1:
        # sum the weights of the first convolution
        weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
                                            axis=1,
                                            keepdim=True)

    model.load_state_dict(weights, strict=False)
    print(f'Loaded {resnet_name} with encoder block {encoder_block} '
          f'pretrained on ImageNet')
    print(ckpt_path)
    return model


if __name__ == '__main__':
    model = ResNet18(block='NonBottleneck1D', pretrained_on_imagenet=True,
                     with_se=True, dilation=[1]*4)

    model.eval()
    print(model)
    image = torch.randn(1, 3, 224, 224)

    from torch.autograd import Variable

    inputs = Variable(image)
    with torch.no_grad():
        outputs = model(inputs)
    for tensor in outputs:
        print(tensor.shape)
