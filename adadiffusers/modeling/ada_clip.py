import torch
import numpy
import os, sys


from torch import nn
from typing import Tuple, Optional, Union

from transformers.models.clip.modeling_clip import (
    CLIPTextEmbeddings,
    CLIPAttention,
    CLIPEncoderLayer,
    CLIPPreTrainedModel,
    CLIPEncoder,
    CLIPTextTransformer,
    CLIPTextModel,
    CLIPVisionEmbeddings,
    CLIPModel,
    CLIPMLP,
    CLIP_START_DOCSTRING,
    CLIP_TEXT_INPUTS_DOCSTRING,
)

from transformers.activations import ACT2FN
from transformers.models.clip.configuration_clip import CLIPTextConfig, CLIPConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

class AdaCLIPTextConfig(CLIPTextConfig):
    def __init__(self, 
                 *args,
                 adapter_emb_size={},
                 use_adapter_attn=True,
                 use_adapter_mlp=True,
                 adapter_act='gelu',
                 adapter_init='normal',
                 adapter_init_std=1e-3,
                 adapter_init_mean=0,
                 use_adapter_norm=True,
                 adapter_pre_norm=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_emb_size = adapter_emb_size
        self.use_adapter_attn = use_adapter_attn
        self.use_adapter_mlp = use_adapter_mlp
        self.adapter_act = adapter_act
        self.adapter_init = adapter_init
        self.adapter_init_mean = adapter_init_mean
        self.adapter_init_std = adapter_init_std 
        self.use_adapter_norm = use_adapter_norm
        self.adapter_pre_norm = adapter_pre_norm


class AdapterLayer(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        self.linear_down = nn.Linear(config.hidden_size, config.adapter_emb_size[layer])
        self.act = ACT2FN[config.adapter_act] if config.adapter_act else None
        self.linear_up = nn.Linear(config.adapter_emb_size[layer], config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.use_adapter_norm else None
                
    def forward(self, hidden_states):
        res = hidden_states
        if self.layernorm is not None and self.config.adapter_pre_norm:
            hidden_states = self.layernorm(hidden_states)
        hidden_states = self.linear_down(hidden_states)
        if self.act is not None:
            hidden_states = self.act(hidden_states)
        hidden_states = self.linear_up(hidden_states)
        if self.layernorm is not None and not self.config.adapter_pre_norm:       
            hidden_states = self.layernorm(hidden_states)
        hidden_states = hidden_states + res
        return hidden_states
    
    
class AdaCLIPEncoderLayer(CLIPEncoderLayer):
    def __init__(self, config, layer):
        super().__init__(config)
        self.config = config
        if config.use_adapter_attn:
            self.adapter_attn = AdapterLayer(config, layer)
        
        if config.use_adapter_mlp:
            self.adapter_mlp = AdapterLayer(config, layer)
            
    def foward(self,
               hidden_states: torch.Tensor,
               attention_mask: torch.Tensor,
               causal_attention_mask: torch.Tensor,
               output_attentions: Optional[bool] = False,
            ) -> Tuple[torch.FloatTensor]:
        
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        if self.config.use_adapter_attn:
            hidden_states = self.adapter_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.config.use_adapter_mlp:
            hidden_states = self.adapter_mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs
    
class AdaCLIPEncoder(CLIPEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([AdaCLIPEncoderLayer(config, layer) for layer in range(config.num_hidden_layers)])

        
class AdaCLIPPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = AdaCLIPTextConfig
    base_model_prefix = "clip"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    
    def _init_adapters(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, AdaCLIPEncoder): # initialize adapter layer with identity projectioin       
            for name, param in module.named_parameters():
                if 'adapter' in name:
                    if 'linear' in name:
                        if 'weight' in name:                            
                            if self.config.adapter_init == 'normal': 
                                # initialize adapter as near-identity function
                                param.data.normal_(mean=self.config.adapter_init_mean, std=self.config.adapter_init_std)
                            elif self.config.adapter_init == 'uniform':
                                param.data.uniform_(a=-self.config.adapter_init_value, b=self.config.adapter_init_value)
                            elif self.config.adapter_init == 'constant':
                                nn.init.constant_(param, self.config.adapter_init_value)
                            elif self.config.adapter_init == 'eye':
                                nn.init.eye_(param)
                            elif self.config.adapter_init == 'zero':
                                param.data.zero_()
                            elif self.config.adapter_init == 'he':
                                nn.init.kaiming_uniform_(param, a=math.sqrt(5)) 
                            else:
                                raise ValueError('error') 
                        elif 'bias' in name:
                            param.data.zero_()
                    else:
                        continue

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, AdaCLIPEncoder):
            module.gradient_checkpointing = value

            
class AdaCLIPTextTransformer(CLIPTextTransformer):
    def __init__(self, config: AdaCLIPTextConfig):
        super().__init__(config)
        self.encoder = AdaCLIPEncoder(config)
    
    
    
class AdaCLIPTextModel(CLIPTextModel, AdaCLIPPreTrainedModel):
    config_class = AdaCLIPTextConfig
    
    def __init__(self, config: AdaCLIPTextConfig):
        super().__init__(config)
        self.text_model = AdaCLIPTextTransformer(config)
        # Initialize adapter
        self._init_adapters(self.text_model.encoder)
