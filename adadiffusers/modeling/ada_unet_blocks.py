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
from .ada_resnet import AdaResnetBlock2D, Upsample2D, Downsample2D, AdaUpsample2D, AdaDownsample2D

from diffusers.models.resnet import (
    # Downsample2D, 
    FirDownsample2D, 
    FirUpsample2D, 
    ResnetBlock2D, 
    # Upsample2D
)

from diffusers.models.unet_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
    DownEncoderBlock2D, 
    UpDecoderBlock2D
)    

class AdaDownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        adapter_config,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                AdaResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    adapter_config=adapter_config,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    AdaDownsample2D(
                        adapter_config,
                        in_channels, 
                        use_conv=True,
                        out_channels=out_channels, 
                        padding=downsample_padding, 
                        name="op", 
                    )
                ]
            )
        else:
            self.downsamplers = None
    
        self.gradient_checkpointing = False
    
    def forward(self, hidden_states):
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, None)
            else:
                hidden_states = resnet(hidden_states, None)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(downsampler), hidden_states)
                else:
                    hidden_states = downsampler(hidden_states)

        return hidden_states



class AdaDownBlock2D(nn.Module):
    def __init__(
        self,
        adapter_config,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                AdaResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    adapter_config=adapter_config,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    AdaDownsample2D(
                        adapter_config,
                        in_channels, 
                        use_conv=True, 
                        out_channels=out_channels,
                        padding=downsample_padding, 
                        name="op", 
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

    
class AdaCrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        adapter_config,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                AdaResnetBlock2D(
                    adapter_config=adapter_config,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                AdaSpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                    adapter_config=adapter_config,
                    dropout=0.0,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    AdaDownsample2D(
                        adapter_config,
                        in_channels, 
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding, 
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.attn_num_head_channels % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        if slice_size is not None and slice_size > self.attn_num_head_channels:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )

        for attn in self.attentions:
            attn._set_attention_slice(slice_size)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn), hidden_states, encoder_hidden_states
                )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, context=encoder_hidden_states)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states    
        

class AdaUpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        adapter_config,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                AdaResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    adapter_config=adapter_config,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([AdaUpsample2D(adapter_config, out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None
        
        self.gradient_checkpointing = False

    def forward(self, hidden_states):
        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, None)
            else:
                hidden_states = resnet(hidden_states, None)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                if self.training and self.gradient_checkpointing:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(upsampler), hidden_states)
                else:
                    hidden_states = upsampler(hidden_states)

        return hidden_states        
        

    
class AdaCrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        adapter_config,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        attention_type="default",
        output_scale_factor=1.0,
        downsample_padding=1,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                AdaResnetBlock2D(
                    adapter_config=adapter_config,
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            attentions.append(
                AdaSpatialTransformer(
                    out_channels,
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                    adapter_config=adapter_config,
                    dropout=0.0,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([AdaUpsample2D(adapter_config, out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.attn_num_head_channels % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        if slice_size is not None and slice_size > self.attn_num_head_channels:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )

        for attn in self.attentions:
            attn._set_attention_slice(slice_size)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn), hidden_states, encoder_hidden_states
                )
            else:
                hidden_states = resnet(hidden_states, temb)
                hidden_states = attn(hidden_states, context=encoder_hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states    
    
class AdaUNetMidBlock2D(nn.Module):
    def __init__(
        self,
        adapter_config,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        attention_type="default",
        output_scale_factor=1.0,
        **kwargs,
    ):
        super().__init__()

        self.attention_type = attention_type
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            AdaResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                adapter_config=adapter_config,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                AttentionBlock(
                    in_channels,
                    num_head_channels=attn_num_head_channels,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    num_groups=resnet_groups,
                )
            )
            resnets.append(
                AdaResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    adapter_config=adapter_config,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        
        self.gradient_checkpointing=False
        
    def forward(self, hidden_states, temb=None, encoder_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                if self.attention_type == "default":
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn), hidden_states)
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(attn), hidden_states, encoder_states)
                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:  
                if self.attention_type == "default":
                    hidden_states = attn(hidden_states)
                else:
                    hidden_states = attn(hidden_states, encoder_states)
                hidden_states = resnet(hidden_states, temb)

        return hidden_states

    
class AdaUNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        adapter_config,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        attention_type="default",
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        **kwargs,
    ):
        super().__init__()

        self.attention_type = attention_type
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            AdaResnetBlock2D(
                adapter_config=adapter_config,
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                AdaSpatialTransformer(
                    in_channels,
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    depth=1,
                    context_dim=cross_attention_dim,
                    num_groups=resnet_groups,
                    adapter_config=adapter_config,
                    dropout=0.0,
                )
            )
            resnets.append(
                AdaResnetBlock2D(
                    adapter_config=adapter_config,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.attn_num_head_channels % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )
        if slice_size is not None and slice_size > self.attn_num_head_channels:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.attn_num_head_channels}"
            )

        for attn in self.attentions:
            attn._set_attention_slice(slice_size)

    def forward(self, hidden_states, temb=None, encoder_hidden_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states

class AdaUpBlock2D(nn.Module):
    def __init__(
        self,
        adapter_config,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                AdaResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    adapter_config=adapter_config,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([AdaUpsample2D(adapter_config, out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
            else:
                hidden_states = resnet(hidden_states, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


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
        if adapter_config.down_block:
            return AdaDownBlock2D(
                num_layers=num_layers,
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                add_downsample=add_downsample,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                downsample_padding=downsample_padding,
                adapter_config=adapter_config,
            )
        else:
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
        if adapter_config.down_block:
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
    elif down_block_type == "DownEncoderBlock2D":
        if adapter_config.down_block:
            return AdaDownEncoderBlock2D(
                num_layers=num_layers,
                in_channels=in_channels,
                out_channels=out_channels,
                add_downsample=add_downsample,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                downsample_padding=downsample_padding,
                adapter_config=adapter_config,
            )
        else:    
            return DownEncoderBlock2D(
                num_layers=num_layers,
                in_channels=in_channels,
                out_channels=out_channels,
                add_downsample=add_downsample,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                downsample_padding=downsample_padding,
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
        if adapter_config.up_block:
            return AdaUpBlock2D(
                num_layers=num_layers,
                in_channels=in_channels,
                out_channels=out_channels,
                prev_output_channel=prev_output_channel,
                temb_channels=temb_channels,
                add_upsample=add_upsample,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                adapter_config=adapter_config,
            )
        else:    
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
        if adapter_config.up_block:
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
    elif up_block_type == "UpDecoderBlock2D":
        if adapter_config.up_block:
            return AdaUpDecoderBlock2D(
                num_layers=num_layers,
                in_channels=in_channels,
                out_channels=out_channels,
                add_upsample=add_upsample,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
                adapter_config=adapter_config,
            )
        else:    
            return UpDecoderBlock2D(
                num_layers=num_layers,
                in_channels=in_channels,
                out_channels=out_channels,
                add_upsample=add_upsample,
                resnet_eps=resnet_eps,
                resnet_act_fn=resnet_act_fn,
                resnet_groups=resnet_groups,
            )
    raise ValueError(f"{up_block_type} does not exist.")
    
