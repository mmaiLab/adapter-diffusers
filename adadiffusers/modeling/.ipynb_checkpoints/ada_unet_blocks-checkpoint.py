import numpy as np

# limitations under the License.
import torch
from torch import nn
import torch.nn.functional as F

from diffusers.models.attention import (
    BasicTransformerBlock,
    AttentionBlock, 
    SpatialTransformer,
)

from .ada_attention import AdaSpatialTransformer

from diffusers.models.resnet import (
    Downsample2D, 
    FirDownsample2D, 
    FirUpsample2D, 
    ResnetBlock2D, 
    Upsample2D
)

from diffusers.models.unet_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
)    
        

class AdaCrossAttnDownBlock2D(CrossAttnDownBlock2D):
    def __init__(self,
                 adapter_config,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        out_channels = kwargs['out_channels']
        num_layers = kwargs['num_layers']
        attn_num_head_channels = kwargs['attn_num_head_channels']
        cross_attention_dim = kwargs['cross_attention_dim']
        resnet_groups = kwargs['resnet_groups']
        
        attentions = []
        for i in range(num_layers):
            attentions.append(
                AdaSpatialTransformer(
                        out_channels,
                        attn_num_head_channels,
                        out_channels // attn_num_head_channels,
                        adapter_config=adapter_config,
                        depth=1,
                        dropout=0.0,
                        context_dim=cross_attention_dim,
                        #num_groups=resnet_groups,
                    )
            )
        self.attentions = nn.ModuleList(attentions)


class AdaCrossAttnUpBlock2D(CrossAttnUpBlock2D):
    def __init__(self,
                 adapter_config,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        out_channels = kwargs['out_channels']
        num_layers = kwargs['num_layers']
        attn_num_head_channels = kwargs['attn_num_head_channels']
        cross_attention_dim = kwargs['cross_attention_dim']
        resnet_groups = kwargs['resnet_groups']
        
        attentions = []
        for i in range(num_layers):
            attentions.append(
                AdaSpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    adapter_config=adapter_config,
                    depth=1,
                    dropout=0.0,
                    context_dim=cross_attention_dim,
                    #num_groups=resnet_groups,
                )
            )
        self.attentions = nn.ModuleList(attentions)

class AdaUNetMidBlock2DCrossAttn(UNetMidBlock2DCrossAttn):
    def __init__(
            self,
            adapter_config,
            *args,
            #adapter_config,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        in_channels = kwargs['in_channels']
        num_layers = kwargs['num_layers']
        attn_num_head_channels = kwargs['attn_num_head_channels']
        cross_attention_dim = kwargs['cross_attention_dim']
        resnet_groups = kwargs['resnet_groups']
        
        attentions = []
        for _ in range(num_layers):
            attentions.append(
                AdaSpatialTransformer(
                    in_channels,
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    adapter_config=adapter_config,
                    depth=1,
                    dropout=0.0,
                    context_dim=cross_attention_dim,
                    #num_groups=resnet_groups,
                )
            )
        self.attentions = nn.ModuleList(attentions)

def get_ada_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    adapter_config,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
        )
    
    elif down_block_type == "CrossAttnDownBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock2D")
        if adapter_config['down_block']:
            return AdaCrossAttnDownBlock2D(
                num_layers=num_layers,
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                add_downsample=add_downsample,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                downsample_padding=downsample_padding,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attn_num_head_channels,
                adapter_config=adapter_config,
            )
        else:
            return CrossAttnDownBlock2D(
                num_layers=num_layers,
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                add_downsample=add_downsample,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                downsample_padding=downsample_padding,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attn_num_head_channels,
            )
    

def get_ada_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    adapter_config,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,    
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
        )
    
    elif up_block_type == "CrossAttnUpBlock2D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock2D")
        if adapter_config['up_block']:
            return AdaCrossAttnUpBlock2D(
                num_layers=num_layers,
                in_channels=in_channels,
                out_channels=out_channels,
                prev_output_channel=prev_output_channel,
                temb_channels=temb_channels,
                add_upsample=add_upsample,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attn_num_head_channels,
                adapter_config=adapter_config,
            )
        else:
            return CrossAttnUpBlock2D(
                num_layers=num_layers,
                in_channels=in_channels,
                out_channels=out_channels,
                prev_output_channel=prev_output_channel,
                temb_channels=temb_channels,
                add_upsample=add_upsample,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attn_num_head_channels,
            )
    
    raise ValueError(f"{up_block_type} does not exist.")
