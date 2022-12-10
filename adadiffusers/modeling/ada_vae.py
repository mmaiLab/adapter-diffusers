from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.models.vae import DecoderOutput, AutoencoderKLOutput, DiagonalGaussianDistribution
from diffusers.models.unet_blocks import DownEncoderBlock2D, UpDecoderBlock2D, UNetMidBlock2D
from .ada_unet_blocks import (
    AdaDownEncoderBlock2D, 
    AdaUpDecoderBlock2D, 
    AdaUNetMidBlock2D, 
    get_ada_down_block, 
    get_ada_up_block,
)


class AdaEncoder(nn.Module):
    def __init__(
        self,
        adapter_config,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = torch.nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        
        emb_size_ratio = adapter_config.adapter_emb_size_ratio
        if not emb_size_ratio:
            adapter_emb_sizes = adapter_config.adapter_emb_sizes['vae_encoder']
        adapter_blocks = adapter_config.vae_blocks
        
        # down
        down_blocks = adapter_blocks['enc_down']
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            adapter_emb_size = adapter_emb_sizes[i]
            if not emb_size_ratio:
                adapter_config.adapter_emb_size = adapter_emb_size
            adapter_config.down_block = down_blocks[i]
            
            down_block = get_ada_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
                adapter_config=adapter_config,
            )
            self.down_blocks.append(down_block)
        
        if not emb_size_ratio:
            adapter_config.adapter_emb_size = adapter_emb_sizes[-1]
        adapter_config.mid_block = adapter_blocks['enc_mid']
        
        # mid
        if adapter_config.mid_block:
            self.mid_block = AdaUNetMidBlock2D(
                in_channels=block_out_channels[-1],
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                output_scale_factor=1,
                resnet_time_scale_shift="default",
                attn_num_head_channels=None,
                resnet_groups=norm_num_groups,
                temb_channels=None,
                adapter_config=adapter_config,
            )
        else:
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                output_scale_factor=1,
                resnet_time_scale_shift="default",
                attn_num_head_channels=None,
                resnet_groups=norm_num_groups,
                temb_channels=None,
            )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

    def forward(self, x):
        sample = x
        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
    

class AdaDecoder(nn.Module):
    def __init__(
        self,
        adapter_config,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        
        emb_size_ratio = adapter_config.adapter_emb_size_ratio
        if not emb_size_ratio:
            adapter_emb_sizes = adapter_config.adapter_emb_sizes['vae_decoder']
        adapter_blocks = adapter_config.vae_blocks
        
        # mid
        if not emb_size_ratio:
            adapter_config.adapter_emb_size = adapter_emb_sizes[-1]
        adapter_config.mid_block = adapter_blocks['dec_mid']
        
        if adapter_config.mid_block:
            self.mid_block = AdaUNetMidBlock2D(
                in_channels=block_out_channels[-1],
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                output_scale_factor=1,
                resnet_time_scale_shift="default",
                attn_num_head_channels=None,
                resnet_groups=norm_num_groups,
                temb_channels=None,
                adapter_config=adapter_config
            )
        else:
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                output_scale_factor=1,
                resnet_time_scale_shift="default",
                attn_num_head_channels=None,
                resnet_groups=norm_num_groups,
                temb_channels=None,
            )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        if not emb_size_ratio:
            reversed_adapter_emb_sizes = list(reversed(adapter_emb_sizes))
        up_blocks = list(reversed(adapter_blocks['dec_up']))
        
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            
            if not emb_size_ratio:
                adapter_config.adapter_emb_size = reversed_adapter_emb_sizes[i]
            adapter_config.up_block = up_blocks[i]
            
            up_block = get_ada_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
                adapter_config=adapter_config,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(self, z):
        sample = z
        sample = self.conv_in(sample)

        # middle
        sample = self.mid_block(sample)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
    

class AdaAutoencoderKL(ModelMixin, ConfigMixin):
    r"""Variational Autoencoder (VAE) model with KL loss from the paper Auto-Encoding Variational Bayes by Diederik P. Kingma
    and Max Welling.
    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)
    Parameters:
        in_channels (int, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (int,  *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownEncoderBlock2D",)`): Tuple of downsample block types.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("UpDecoderBlock2D",)`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(64,)`): Tuple of block output channels.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        latent_channels (`int`, *optional*, defaults to `4`): Number of channels in the latent space.
        sample_size (`int`, *optional*, defaults to `32`): TODO
    """

    @register_to_config
    def __init__(
        self,
        adapter_config,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = AdaEncoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            adapter_config=adapter_config,
        )

        # pass init params to Decoder
        self.decoder = AdaDecoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            adapter_config=adapter_config,
        )

        self.quant_conv = torch.nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = torch.nn.Conv2d(latent_channels, latent_channels, 1)
        
        self._supports_gradient_checkpointing = True
        
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (AdaDownEncoderBlock2D, AdaUpDecoderBlock2D, AdaUNetMidBlock2D)):
            module.gradient_checkpointing = value

    def encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    
    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)