# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from modulefinder import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import re
class ConvBNAct(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size,
                 activation=nn.ReLU(inplace=True), dilation=1, stride=1):
        super(ConvBNAct, self).__init__()
        padding = kernel_size // 2 + dilation - 1
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          bias=False,
                                          dilation=dilation,
                                          stride=stride))
        self.add_module('bn', nn.BatchNorm2d(channels_out))
        self.add_module('act', activation)


class ConvBN(nn.Sequential):
    def __init__(self, channels_in, channels_out, kernel_size):
        super(ConvBN, self).__init__()
        self.add_module('conv', nn.Conv2d(channels_in, channels_out,
                                          kernel_size=kernel_size,
                                          padding=kernel_size // 2,
                                          bias=False))
        self.add_module('bn', nn.BatchNorm2d(channels_out))


class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y


class SqueezeAndExcitationTensorRT(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitationTensorRT, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # TensorRT restricts the maximum kernel size for pooling operations
        # by "MAX_KERNEL_DIMS_PRODUCT" which leads to problems if the input
        # feature maps are of large spatial size
        # -> workaround: use cascaded two-staged pooling
        # see: https://github.com/onnx/onnx-tensorrt/issues/333
        if x.shape[2] > 120 and x.shape[3] > 160:
            weighting = F.adaptive_avg_pool2d(x, 4)
        else:
            weighting = x
        weighting = F.adaptive_avg_pool2d(weighting, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y

#*SpatialGroupEnhance
class SpatialGroupEnhance(nn.Module):
    '''
        Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks
        https://arxiv.org/pdf/1905.09646.pdf
    '''
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

#*Triplet Attention
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


# class SpatialGate(nn.Module):
#     def __init__(self):
#         super(SpatialGate, self).__init__()

#         self.channel_pool = ChannelPool()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
#             nn.BatchNorm2d(1)
#         )
#         self.sigmod = nn.Sigmoid()

#     def forward(self, x):
#         out = self.conv(self.channel_pool(x))
#         return out * self.sigmod(out)


class TripletAttention(nn.Module):
    def __init__(self, spatial=True):
        super(TripletAttention, self).__init__()
        self.spatial = spatial
        self.height_gate = SpatialGate()
        self.width_gate = SpatialGate()
        if self.spatial:
            self.spatial_gate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.height_gate(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.width_gate(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()

        if self.spatial:
            x_out3 = self.spatial_gate(x)
            return (1/3) * (x_out1 + x_out2 + x_out3)
        else:
            return (1/2) * (x_out1 + x_out2)



#*CBAM
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16,activation=nn.ReLU(inplace=True)):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            activation,
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    """
        CBAM: Convolutional Block Attention Module
        https://arxiv.org/pdf/1807.06521.pdf
    """
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

#*BAM
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_activation =nn.ReLU(inplace=True)
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor


#*SMRLayer
class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.cfc = Parameter(torch.Tensor(channel, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()

        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps
        channel_std = channel_var.sqrt()

        t = torch.cat((channel_mean, channel_std), dim=2)
        return t 
    
    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2
        z = torch.sum(z, dim=2)[:, :, None, None] # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)

        return x * g


# *    channel代表in_channels  # *sSE 
class ChSqueezeAndSpExcitation(nn.Module):
    def __init__(self, channel):
        super(ChSqueezeAndSpExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = self.fc(x)
        y=weighting*x
        return y
 
 
   # *cSE  
class SpSqueezeAndChExcitation(nn.Module):
    def __init__(self, channel):
        super(SpSqueezeAndChExcitation, self).__init__()
        self.fc = nn.Sequential(
            #第一次全连接，降低维度
            nn.Conv2d(channel, channel // 2, kernel_size=1, bias=False),
            #第二次全连接，恢复维度
            nn.Conv2d(channel // 2, channel, kernel_size=1, bias=False), 
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = F.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y   
    
 #*scSE
class scSE(nn.Module):
    def __init__(self, channel):
        super(scSE,self).__init__()
        self.sSE=ChSqueezeAndSpExcitation(channel)
        self.cSE=SpSqueezeAndChExcitation(channel)
        
    def forward(self,x):
        y = torch.max(self.cSE(x), self.sSE(x))
        return y

# *ECA
class ESqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(ESqueezeAndExcitation, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Conv1d(1, 1 kernel_size=3,padding=(3 - 1) // 2, bias=False),
        #     activation,
        #     nn.Sigmoid()
        # )
        
        # *kernel_size=3,5,7,9,11  resnet越小，ksize越大
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3- 1) // 2, bias=False) 
        self.activation=activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # # feature descriptor on the global spatial information
        # y = F.adaptive_avg_pool2d(x, 1)
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y=self.activation(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# *SPA_124
# class  SPALayer(nn.Module):
#     def __init__(self,channel, reduction=16,activation=nn.ReLU(inplace=True)):
#         super(SPALayer, self).__init__()
#         self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
#         self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
#         self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
#         self.weight = Parameter(torch.ones(1, 3, 1, 1, 1))
#         self.transform = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.BatchNorm2d(channel // reduction),
#             activation,
#             nn.Conv2d(channel // reduction, channel, 1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y1 = self.avg_pool1(x)
#         y2 = self.avg_pool2(x)
#         y4 = self.avg_pool4(x)
#         y = torch.cat(
#             [y4.unsqueeze(dim=1),
#              F.interpolate(y2, scale_factor=2).unsqueeze(dim=1),
#              F.interpolate(y1, scale_factor=4).unsqueeze(dim=1)],
#             dim=1
#         )
#         y = (y * self.weight).sum(dim=1, keepdim=False)
#         y = self.transform(y)
#         y = F.interpolate(y, size=x.size()[2:])

#         return x * y
    
# *SPA_B
class SPABLayer(nn.Module):
    def __init__(self, inchannel,channel, reduction=16,activation=nn.ReLU(inplace=True)):
        super(SPABLayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.weight = Parameter(torch.ones(1,3,1,1,1))
        self.transform = nn.Sequential(
            nn.Conv2d(inchannel, channel//reduction,1,bias=False),
            nn.BatchNorm2d(channel//reduction),
            activation,
            nn.Conv2d(channel//reduction, channel, 1,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c,_, _ = x.size()
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool2(x)
        y4 = self.avg_pool4(x)
        y = torch.cat(
            [y4.unsqueeze(dim=1),
             F.interpolate(y2,scale_factor=2).unsqueeze(dim=1),
             F.interpolate(y1,scale_factor=4).unsqueeze(dim=1)],
            dim=1
        )
        y = (y*self.weight).sum(dim=1,keepdim=False)
        y = self.transform(y)

        return y

# *SPA_C
class SPACLayer(nn.Module):
    def __init__(self, inchannel,channel, reduction=16,activation=nn.ReLU(inplace=True)):
        super(SPACLayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.weight = Parameter(torch.ones(1,3,1,1,1))
        if inchannel !=channel:
            self.matcher = nn.Sequential(
                nn.Conv2d(inchannel, channel//reduction,1,bias=False),
                nn.BatchNorm2d(channel//reduction),
                activation,
                nn.Conv2d(channel//reduction, channel, 1,bias=False),
                nn.BatchNorm2d(channel)
            )
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel//reduction,1,bias=False),
            nn.BatchNorm2d(channel//reduction),
            activation,
            nn.Conv2d(channel//reduction, channel, 1,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.matcher(x) if hasattr(self, 'matcher') else x
        b, c,_, _ = x.size()
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool2(x)
        y4 = self.avg_pool4(x)
        y = torch.cat(
            [y4.unsqueeze(dim=1),
             F.interpolate(y2,scale_factor=2).unsqueeze(dim=1),
             F.interpolate(y1,scale_factor=4).unsqueeze(dim=1)],
            dim=1
        )
        y = (y*self.weight).sum(dim=1,keepdim=False)
        y = self.transform(y)

        return y

# # *SPA_147
class SPALayer(nn.Module):
    def __init__(self,channel, reduction=16,activation=nn.ReLU(inplace=True)):
        super(SPALayer, self).__init__()
        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
        self.avg_pool7 = nn.AdaptiveAvgPool2d(7)
        self.weight = Parameter(torch.ones(1, 3, 1, 1, 1))
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            activation,
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool1(x)
        y2 = self.avg_pool4(x)
        y4 = self.avg_pool7(x)
        y = torch.cat(
            [y4.unsqueeze(dim=1),
             F.interpolate(y2, size=[7,7]).unsqueeze(dim=1),
             F.interpolate(y1, size=[7,7]).unsqueeze(dim=1)],
            dim=1
        )
        y = (y * self.weight).sum(dim=1, keepdim=False)
        y = self.transform(y)
        y = F.interpolate(y, size=x.size()[2:])

        return x * y
 
#  #*SPAD147
# class SPADLayer(nn.Module):
#     def __init__(self,channel, reduction=16,activation=nn.ReLU(inplace=True),k_size=9):
#         super(SPADLayer, self).__init__()
#         self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
#         self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
#         self.avg_pool7 = nn.AdaptiveAvgPool2d(7)
#         self.weight = Parameter(torch.ones(1, 3,  1, 1))
#         self.conv = nn.Conv2d(1, 256, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.transform = nn.Sequential(
#             # nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             # nn.BatchNorm2d(channel // reduction),
#             # activation,
#             # nn.Conv2d(channel // reduction, channel, 1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y1 = self.avg_pool1(x)
#         y2 = self.avg_pool4(x)
#         y4 = self.avg_pool7(x)
#         y = torch.cat(
#             [y4.unsqueeze(dim=1),
#              F.interpolate(y2, size=[7,7]).unsqueeze(dim=1),
#              F.interpolate(y1, size=[7,7]).unsqueeze(dim=1)],
#             dim=1
#         )
    
#         y = y.sum(dim=1, keepdim=False)
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         y = self.transform(y)
#         # y = F.interpolate(y, size=x.size()[2:])

#         return x * y.expand_as(x)
 
#  # *SPA_1247   
# class SPALayer(nn.Module):
#     def __init__(self,channel, reduction=16,activation=nn.ReLU(inplace=True)):
#         super(SPALayer, self).__init__()
#         self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
#         self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
#         self.avg_pool4 = nn.AdaptiveAvgPool2d(4)
#         self.avg_pool7 = nn.AdaptiveAvgPool2d(7)

#         self.weight = Parameter(torch.ones(1, 4, 1, 1, 1))
#         self.transform = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.BatchNorm2d(channel // reduction),
#             activation,
#             nn.Conv2d(channel // reduction, channel, 1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y1 = self.avg_pool1(x)
#         y2 = self.avg_pool2(x)
#         y4 = self.avg_pool4(x)
#         y7 = self.avg_pool7(x)
#         y = torch.cat(
#             [y7.unsqueeze(dim=1),
#              F.interpolate(y4, size=[7, 7]).unsqueeze(dim=1),
#              F.interpolate(y2, size=[7, 7]).unsqueeze(dim=1),
#              F.interpolate(y1, size=[7, 7]).unsqueeze(dim=1)],
#             dim=1
#         )
#         y = (y * self.weight).sum(dim=1, keepdim=False)
#         y = self.transform(y)
#         y = F.interpolate(y, size=x.size()[2:])

#         return x * y
    
# *SPA_12
# class SPALayer(nn.Module):
#     def __init__(self,channel, reduction=16,activation=nn.ReLU(inplace=True)):
#         super(SPALayer, self).__init__()
#         self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
#         self.avg_pool2 = nn.AdaptiveAvgPool2d(2)

#         self.weight = Parameter(torch.ones(1, 2, 1, 1, 1))
#         self.transform = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, bias=False),
#             nn.BatchNorm2d(channel // reduction),
#             activation,
#             nn.Conv2d(channel // reduction, channel, 1, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y1 = self.avg_pool1(x)
#         y2 = self.avg_pool2(x)
#         y = torch.cat(
#             [y2.unsqueeze(dim=1),
#              F.interpolate(y1, size=[2,2]).unsqueeze(dim=1)],
#             dim=1
#         )
#         y = (y * self.weight).sum(dim=1, keepdim=False)
#         y = self.transform(y)
#         y = F.interpolate(y, size=x.size()[2:])

#         return x * y
  

class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


def swish(x):
    return x * torch.sigmoid(x)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.
