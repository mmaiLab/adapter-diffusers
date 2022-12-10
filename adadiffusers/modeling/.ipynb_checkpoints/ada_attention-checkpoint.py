import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


from diffusers.models.attention import (
    BasicTransformerBlock,
    AttentionBlock, 
    SpatialTransformer,
)

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

ACT2FN = {
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "silu": nn.SiLU(),
    "swish": lambda x: F.silu(x),
    'mish': Mish(),
}


class AdapterModule(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 emb_size, 
                 init_std=1.0e-3, 
                 norm_type='layer', 
                 act='relu',
                 preln=True):
        super().__init__()
        self.preln = preln
        self.linear_down = nn.Linear(hidden_size, emb_size)
        self.act = ACT2FN[act]
        self.linear_up = nn.Linear(emb_size, hidden_size)
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(hidden_size, eps=1.0e-6)
        
        ## init adapter near identity-function
        for module in [self.linear_down, self.linear_up]:
            for name, param in module.named_parameters():
                if 'bias' in name:
                    param.data.zero_()
                elif 'weight' in name:
                    param.data.normal_(mean=0, std=init_std)
                else:
                    raise ValueError(f'{name} param does not exist.')
    
        
    def forward(self, hidden_states):
        res = hidden_states
        if self.preln:
            hidden_states = self.norm(hidden_states)
        hidden_states = self.linear_down(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_up(hidden_states)
        if not self.preln:
            hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states + res
        return hidden_states
    

    
class AdaSpatialTransformer(SpatialTransformer):
    def __init__(self,
                 *args,
                 adapter_config,
                 **kwargs):
        super().__init__(*args, **kwargs)
        depth = kwargs['depth']
        dropout = kwargs['dropout']
        context_dim = kwargs['context_dim']
        inner_dim = self.n_heads*self.d_head
        
        # resolve adapter_config
        modules = adapter_config['modules']
        adapter_emb_size = adapter_config['adapter_emb_size']
        adapter_act = adapter_config['adapter_act']
        skip_adapter = adapter_config['skip_adapter']
        preln = adapter_config['preln']
        
        self.transformer_blocks = nn.ModuleList(
            [
                AdaTransformerBlock(
                    inner_dim, 
                    self.n_heads, 
                    self.d_head, 
                    modules=modules, 
                    adapter_emb_size=adapter_emb_size, 
                    adapter_act=adapter_act,
                    skip_adapter=skip_adapter,
                    preln=preln,
                    dropout=dropout, 
                    context_dim=context_dim) for d in range(depth)
            ]
        )
        

class AdaTransformerBlock(BasicTransformerBlock):
    def __init__(self, 
                 *args, 
                 modules, 
                 adapter_emb_size, 
                 adapter_act, 
                 skip_adapter,
                 preln,
                 **kwargs):
        super().__init__(*args, **kwargs)
        dim = args[0]
        context_dim = kwargs['context_dim']
        self.skip_adapter = skip_adapter
        self.modules = modules
        self.adapters = nn.ModuleDict({module: AdapterModule(
            hidden_size = dim, emb_size=adapter_emb_size, act=adapter_act, preln=preln
        ) for module in modules})
        if 'context' in modules:
            self.adapters.update({'context': AdapterModule(hidden_size=context_dim, emb_size=adapter_emb_size, act=adapter_act)})
        
    def forward(self, hidden_states, context=None):
        hidden_states = hidden_states.contiguous() if hidden_states.device.type == "mps" else hidden_states
        if self.skip_adapter:
            res = hidden_states
            hidden_states = self.attn1(self.norm1(hidden_states))
            if 'MHSA' in self.modules:
                hidden_states = self.adapters['MHSA'](hidden_states) + res
            else:
                hidden_states = hidden_states + res
            if 'context' in self.modules:
                context = self.adapters['context'](context)
            res = hidden_states
            hidden_states = self.attn2(self.norm2(hidden_states), context=context)
            if 'MHCA' in self.modules:
                hidden_states = self.adapters['MHCA'](hidden_states) + res
            else:
                hidden_states = hidden_states + res
            res = hidden_states
            hidden_states = self.ff(self.norm3(hidden_states))
            if 'FF' in self.modules:
                hidden_states = self.adapters['FF'](hidden_states) + res
            else:
                hidden_states = hidden_states + res
        else:
            hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
            if 'MHSA' in self.modules:
                hidden_states = self.adapters['MHSA'](hidden_states)
            if 'context' in self.modules:
                context = self.adapters['context'](context)
            hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states
            if 'MHCA' in self.modules:
                hidden_states = self.adapters['MHCA'](hidden_states)
            hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
            if 'FF' in self.modules:
                hidden_states = self.adapters['FF'](hidden_states)
        return hidden_states

        



        

