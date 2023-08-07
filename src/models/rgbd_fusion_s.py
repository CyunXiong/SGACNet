# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch.nn as nn

from src.models.model_utils import SqueezeAndExcitation ,scSE,ChSqueezeAndSpExcitation,SpSqueezeAndChExcitation,ESqueezeAndExcitation,SPALayer,SPABLayer,SPACLayer


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        # self.se_rgb = SqueezeAndExcitation(channels_in,
        #                                    activation=activation)
        # self.se_depth = SqueezeAndExcitation(channels_in,
        #                                      activation=activation)
        
        # *sSE 
        # self.se_rgb = ChSqueezeAndSpExcitation(channels_in)
        # self.se_depth = ChSqueezeAndSpExcitation(channels_in)
        
    #    # *scSE
        # self.se_rgb = scSE(channels_in)
        # self.se_depth = scSE(channels_in)
        
        
        # # *cSE
        # self.se_rgb = SpSqueezeAndChExcitation(channels_in)
        # self.se_depth = SpSqueezeAndChExcitation(channels_in)
         
           # *SPALayer
        self.se_rgb = SPALayer(channels_in,
                                activation=activation)
        self.se_depth =SPALayer(channels_in,
                                activation=activation) 
        
          # *SPABLayer
        self.se_rgb = SPABLayer(channels_in,
                                activation=activation)
        self.se_depth =SPABLayer(channels_in,
                                activation=activation) 
          # *SPACLayer
        self.se_rgb = SPACLayer(channels_in,
                                activation=activation)
        self.se_depth =SPACLayer(channels_in,
                                activation=activation) 
        

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out
    
class ESqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(ESqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = ESqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = ESqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out
