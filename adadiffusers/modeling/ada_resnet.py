import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


from diffusers.models.resnet import ResnetBlock2D

from .adapter import ACT2FN, CNNAdapter
    
class AdaResnetBlock2D(ResnetBlock2D):
    def __init__(self, adapter_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.modules = adapter_config.modules['resnet']
        emb_size_ratio = adapter_config.adapter_emb_size_ratio 
        self.conv1_adapter = CNNAdapter(self.out_channels,
                                        int(self.out_channels*emb_size_ratio) if emb_size_ratio else adapter_config.adapter_emb_size,
                                        act=adapter_config.conv_adapter_act) if 'conv1' in self.modules else None
        
        self.conv2_adapter = CNNAdapter(self.out_channels,                                                                     int(self.out_channels*emb_size_ratio) if emb_size_ratio else adapter_config.adapter_emb_size,
                                           act=adapter_config.conv_adapter_act) if 'conv2' in self.modules else None
        
        
    def forward(self, x, temb):
        hidden_states = x

        # make sure hidden states is in float32
        # when running in half-precision
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            x = self.upsample(x)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            x = self.downsample(x)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)
        
        
        if self.conv1_adapter is not None:
            hidden_states = self.conv1_adapter(hidden_states)
        
        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            hidden_states = hidden_states + temb

        # make sure hidden states is in float32
        # when running in half-precision
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        
        if self.conv2_adapter is not None:
            hidden_states = self.conv2_adapter(hidden_states)
        
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)

        out = (x + hidden_states) / self.output_scale_factor

        return out

        
        
class Upsample2D(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=1)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)
        
        return hidden_states

    
class AdaUpsample2D(Upsample2D):
    def __init__(self, adapter_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.modules = adapter_config.modules
        emb_size_ratio = adapter_config.adapter_emb_size_ratio
        self.upsampler_adapter = CNNAdapter(self.out_channels, 
                                           int(self.out_channels*emb_size_ratio) if emb_size_ratio else adapter_config.adapter_emb_size,
                                           act=adapter_config.conv_adapter_act) if 'upsampler' in self.modules else None 
    
    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)
        
        if self.upsampler_adapter is not None:
            hidden_states = self.upsampler_adapter(hidden_states)
        

        return hidden_states

class Downsample2D(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs. :param use_conv: a bool determining if a convolution is
    applied. :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states

    
class AdaDownsample2D(Downsample2D):
    def __init__(self, adapter_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.modules = adapter_config.modules
        emb_size_ratio = adapter_config.adapter_emb_size_ratio
        self.downsampler_adapter = CNNAdapter(self.out_channels, 
                                           int(self.out_channels*emb_size_ratio) if emb_size_ratio else adapter_config.adapter_emb_size,
                                           act=adapter_config.conv_adapter_act) if 'upsampler' in self.modules else None
    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        if self.downsampler_adapter is not None:
            hidden_states = self.downsampler_adapter(hidden_states)

        return hidden_states



        

