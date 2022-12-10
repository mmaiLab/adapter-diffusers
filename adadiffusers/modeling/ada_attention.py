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
    CrossAttention,
)

from .adapter import ACT2FN, TransformerAdapter

    
class AdaCrossAttention(CrossAttention):
    def __init__(self, adapter_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        inner_dim = kwargs['dim_head'] * kwargs['heads']
        context_dim = kwargs['context_dim']
        query_dim = kwargs['query_dim']
        context_dim = context_dim if context_dim is not None else query_dim
        self.modules = adapter_config.add_modules
        ratio = adapter_config.adapter_emb_size_ratio
        
        if 'cross_attn_context' in self.modules:
            self.adapter_k = TransformerAdapter(context_dim, 
                                         adapter_config.context_adapter_emb_size,
                                         act=adapter_config.context_adapter_act,
                                         preln=adapter_config.preln,
                                         useln=adapter_config.useln,
                                        )
            self.adapter_v = TransformerAdapter(context_dim, 
                                 adapter_config.context_adapter_emb_size,
                                 act=adapter_config.context_adapter_act,
                                 preln=adapter_config.preln,
                                 useln=adapter_config.useln,
                                )
        else: 
            self.adapter_q = TransformerAdapter(inner_dim, 
                                             int(inner_dim*ratio) if ratio else adapter_config.adapter_emb_size,
                                             act=adapter_config.adapter_act,
                                             preln=adapter_config.preln,
                                             useln=adapter_config.useln,
                                            )
            self.adapter_k = TransformerAdapter(inner_dim, 
                                     int(inner_dim*ratio) if ratio else adapter_config.adapter_emb_size,
                                     act=adapter_config.adapter_act,
                                     preln=adapter_config.preln,
                                     useln=adapter_config.useln,
                                    )
            self.adapter_v = TransformerAdapter(inner_dim, 
                                     int(inner_dim*ratio) if ratio else adapter_config.adapter_emb_size,
                                     act=adapter_config.adapter_act,
                                     preln=adapter_config.preln,
                                     useln=adapter_config.useln,
                                    )

        
    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        
        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        if 'cross_attn_context' in self.modules:
            key = self.adapter_k(context)
            value = self.adapter_v(context)
            key = self.to_k(key)
            value = self.to_v(value)
        else:
            key = self.to_k(context)
            value = self.to_v(context)
        
        # adapter
        if 'cross_attn' in self.modules:
            query = self.adapter_q(query)
            key = self.adapter_k(key)
            value = self.adapter_v(value)
        
        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of

        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        return self.to_out(hidden_states)
        
    
    
    
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
        

        self.transformer_blocks = nn.ModuleList(
            [
                AdaTransformerBlock(
                    inner_dim, 
                    self.n_heads, 
                    self.d_head, 
                    adapter_config=adapter_config,
                    dropout=dropout, 
                    context_dim=context_dim) for d in range(depth)
            ]
        )
        

class AdaTransformerBlock(BasicTransformerBlock):
    def __init__(self, 
                 *args, 
                 adapter_config,
                 **kwargs):
        super().__init__(*args, **kwargs)
        dim = args[0]
        n_heads = args[1]
        d_head = args[2]
        dropout = kwargs['dropout']
        context_dim = kwargs['context_dim']
        self.skip_adapter = adapter_config.skip_adapter
        self.modules = adapter_config.modules['transformer']
        self.add_modules = adapter_config.add_modules
        self.add_module_block = adapter_config.add_module_block
        ratio = adapter_config.adapter_emb_size_ratio
        
        if self.modules:
            self.adapters = nn.ModuleDict({module: TransformerAdapter(
                hidden_size = dim, 
                emb_size= int(dim*ratio) if ratio else adapter_config.adapter_emb_size, 
                act=adapter_config.adapter_act, 
                preln=adapter_config.preln,
                useln=adapter_config.useln,
            ) for module in self.modules})
        
        if 'context' in self.add_modules and self.add_module_block:
            self.context_adapter = TransformerAdapter(
                hidden_size=context_dim,
                emb_size=adapter_config.context_adapter_emb_size,
                act=adapter_config.context_adapter_act,
                preln=adapter_config.preln,
                useln=adapter_config.useln,
            )
            
        if ('cross_attn' in self.add_modules or 'cross_attn_context' in self.add_modules) and adapter_config.add_module_block:
            self.attn2 = AdaCrossAttention(
                adapter_config=adapter_config,
                query_dim=dim, 
                context_dim=context_dim,
                heads=n_heads, 
                dim_head=d_head, 
                dropout=dropout
            )
        
#         if self.brend:
#             self.alpha = nn.Paramter(torch.tensor([1.]))
#             self.adapter_brend_instance = TransformerAdapter(
#                 hidden_size = dim, 
#                 emb_size=adapter_config.adapter_emb_size, 
#                 act=adapter_config.adapter_act, 
#                 preln=adapter_config.preln,
#                 useln=adapter_config.useln,
#             )
#             self.adapter_brend_prior = TransformerAdapter(
#                 hidden_size = dim, 
#                 emb_size=adapter_config.adapter_emb_size, 
#                 act=adapter_config.adapter_act, 
#                 preln=adapter_config.preln,
#                 useln=adapter_config.useln,
#             )
        
    def forward(self, hidden_states, context=None):
        hidden_states = hidden_states.contiguous() if hidden_states.device.type == "mps" else hidden_states
        if self.skip_adapter:
            res = hidden_states
            hidden_states = self.attn1(self.norm1(hidden_states))
            if 'MHSA' in self.modules:
                hidden_states = self.adapters['MHSA'](hidden_states) + res
            else:
                hidden_states = hidden_states + res
            if 'context' in self.add_modules and self.add_module_block:
                context = self.context_adapter(context)
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
#         elif self.brend:
#             res = hidden_states
#             hidden_states = self.attn1(self.norm1(hidden_states)) + res
#             res = hidden_states
#             hidden_states = self.attn2(self.norm2(hidden_states), context=context) + res
#             res = hidden_states
#             hidden_states_instance, hidden_states_prior = hidden_states.chunk(2)
#             hidden_states_instance = self.adapter_brend_instance(hidden_states_instance)
#             hidden_states_prior = self.adapter_brend_prior(hidden_states_prior)
#             hidden_states = 
#             hidden_states = torch.cat([hidden_states_instance, hidden_states_prior])
        else:
            if self.modules:
                hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
                if 'MHSA' in self.modules:
                    hidden_states = self.adapters['MHSA'](hidden_states)
                if 'context' in self.add_modules and self.add_module_block:
                    context = self.context_adapter(context)
                hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states
                if 'MHCA' in self.modules:
                    hidden_states = self.adapters['MHCA'](hidden_states)
                hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
                if 'FF' in self.modules:
                    hidden_states = self.adapters['FF'](hidden_states)
            else:
                hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
                if 'context' in self.add_modules and self.add_module_block:
                    context = self.context_adapter(context)
                hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states
                hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states

        

