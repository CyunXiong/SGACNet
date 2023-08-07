"""MobileNet and MobileNetV2."""
from pickle import TRUE
from turtle import forward
from numpy import True_
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.hub import load_state_dict_from_url 
from torchvision.models.mobilenet import ConvBNReLU
import torchvision
__all__ = ['MobileNetV2',  'mobilenet_v2']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
}

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
 
       # *可能新增 dilation=None, width_per_group=64,norm_layer=None,norm_layer=nn.BatchNorm2d
class MobileNetV2(nn.Module):
    def __init__(self,layrs,dilation=None, width_per_group=64,num_classes=1000, width_mult=1.0, inverted_residual_setting=None,norm_layer=nn.BatchNorm2d,round_nearest=8):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.dilation = 1 #* dilation为增加的参数
        groups=1
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 64
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]#*增加了norm_layer=norm_layer
        
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))#*增加了norm_layer=norm_layer
        # features.append(_ConvBNReLU(input_channels, last_channels, 1, relu6=True, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
   
    #* dilation为增加的参数
        # if dilation is not None:
        #     dilation = dilation
        #     if len(dilation) != 4:
        #         raise ValueError("dilation should be None "
        #                          "or a 4-element tuple, got "
        #                          "{}".format(dilation))
        # else:
        #     dilation = [1, 1, 1, 1]
   
        self.down_2_channels_out = 64
        self.down_4_channels_out = 64 
        self.down_8_channels_out = 128 
        self.down_16_channels_out = 256 
        self.down_32_channels_out = 512 
        
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = norm_layer(64)
        # self.act = nn.ReLU(inplace=True)  
        
        self.layer1 = self._make_layer(
            block, 64, layers[0], dilate=dilation[0] #?layers[0]是blocs_num
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1],
            stride=2, dilate=dilation[1]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2],
            stride=2, dilate=dilation[2]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3],
            stride=2, dilate=dilation[3]
        )
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=1):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
            #?block.expansion找不到expansion
        if stride != 1 or self.inplanes != planes :
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes , stride),
                norm_layer(planes ),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        # return x
        
        # def forward(self, input):
        #     x=ConvBNReLU(input)
        #     x = self.features(x)
        #     x_down2 = self.act(x)
        #     x = self.maxpool(x_down2)

        x_layer1 = self.forward_resblock(x, self.layer1)
        x_layer2 = self.forward_resblock(x_layer1, self.layer2)
        x_layer3 = self.forward_resblock(x_layer2, self.layer3)
        x_layer4 = self.forward_resblock(x_layer3, self.layer4)
       

        self.skip3_channels = x_layer3.size()[1]
        self.skip2_channels = x_layer2.size()[1]
        self.skip1_channels = x_layer1.size()[1]
    
        return x
    
    def forward_resblock(self, x, layers):
        for l in layers:
            x = l(x)
        return x
    
    def forward_first_conv(self, x):
        # be aware that maxpool still needs to be applied after this function
        # and before forward_layer1()
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.act(x)  
        x=ConvBNReLU( 3, 64,kernel_size=7, stride=1, groups=1)#*in_channels, out_channels，不知道这么改行不行
        return x
  
    def forward_layer1(self, x):
        # be ware that maxpool still needs to be applied after
        # forward_first_conv() and before this function
        x=self.forward_resblock(x,self.layer1)
        self.skip1_channels = x.size()[1]
        return x  
     
    def forward_layer2(self, x):
        x = self.forward_resblock(x,self.layer2)
        self.skip2_channels = x.size()[1]
        return x

    def forward_layer3(self, x):
        x = self.forward_resblock(x,self.layer3)
        self.skip3_channels = x.size()[1]
        return x

    def forward_layer4(self, x):
        x = self.forward_resblock( x,self.layer4)
        return x                     

def mobilenet_v2(multiplier=1.0, pretrained_on_imagenet=False,progress=True,pretrained_dir='./trained_models/imagenet', **kwargs):
    model = MobileNetV2( **kwargs)
    
    if pretrained_on_imagenet:
        # weights = model_zoo.load_url(model_urls['mobilenet_v2'], model_dir='./')#*model_urls自己在前面导入预训练模型的网址
        # if 'input_channels' in kwargs and kwargs['input_channels'] == 1:
        #     # sum the weights of the first convolution
        #     weights['conv1.weight'] = torch.sum(weights['conv1.weight'],
        #                                         axis=1, keepdim=True)
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        # model.load_state_dict(weights, strict=True)
        print('Loaded mobilenet_v2 pretrained on ImageNet')
      
    # return get_mobilenet_v2(1.0, **kwargs)
    return model



# def mobilenet1_0(**kwargs):
#     return get_mobilenet(1.0, **kwargs)


# # def mobilenet_v2_1_0(**kwargs):
# #     return get_mobilenet_v2(1.0, **kwargs)


# def mobilenet0_75(**kwargs):
#     return get_mobilenet(0.75, **kwargs)


# def mobilenet_v2_0_75(**kwargs):
#     return get_mobilenet_v2(0.75, **kwargs)


# def mobilenet0_5(**kwargs):
#     return get_mobilenet(0.5, **kwargs)


# def mobilenet_v2_0_5(**kwargs):
#     return get_mobilenet_v2(0.5, **kwargs)


# def mobilenet0_25(**kwargs):
#     return get_mobilenet(0.25, **kwargs)


# def mobilenet_v2_0_25(**kwargs):
#     return get_mobilenet_v2(0.25, **kwargs)


if __name__ == '__main__':
    model = mobilenet_v2()
