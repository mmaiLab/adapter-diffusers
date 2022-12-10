import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from typing import Union

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

@dataclass
class AdapterConfig:
    modules: dict = field(default_factory=lambda: {'transformer': ['MHSA', 'MHCA', 'FF'],
                                                   'resnet': ['conv1', 'conv2', 'downsampler', 'upsampler']})
    #adapter_emb_sizes: list = field(default_factory=lambda: [64, 128, 256, 256])
    adapter_emb_size_ratio: Union[None, float] = None
    adapter_emb_sizes: Union[None, dict] = field(default_factory=lambda: {'unet': [64, 128, 256, 256],
                                                             'vae_encoder': [64, 64, 128, 128],
                                                             'vae_decoder': [64, 128, 256, 256]})
    adapter_act: str = 'relu'
    conv_adapter_act: str = 'relu'
    context_adapter_act: str = 'relu'
    preln: bool = False
    useln: bool = False
    skip_adapter: bool = False
    use_residual_adapter: bool = False
    blocks: dict = field(default_factory=lambda: {
                     'down': [True, True, True, True], # CrossAttn, CrossAttn, CrossAttn, Down 
                     'mid': True, 
                     'up': [True, True, True, True] # # CrossAttn, CrossAttn, CrossAttn, Up
                })
    add_modules: list = field(default_factory=lambda: ['context', 'cross_attn', 'cross_attn_context'])
    add_modules_blocks: dict = field(default_factory=lambda: {
                     'down': [False]*4, # CrossAttn, CrossAttn, CrossAttn, Down 
                     'mid': False, 
                     'up': [False]*4 # # CrossAttn, CrossAttn, CrossAttn, Up
                })
    vae_blocks: Union[None, dict] = field(default_factory=lambda: {
                     'enc_down': [True, True, True, True], # DownEncoder, DownEncoder, DownEncoder, DownEncoder
                     'enc_mid': True, 
                     'dec_mid': True,
                     'dec_up': [True, True, True, True] # # UpDecoder, UpDecoder, UpDeocder, UpDecoder
                })
    
    adapter_emb_size: int = 128
    context_adapter_emb_size: int = 256
    up_block: bool = False
    down_block: bool = False
    add_module_block: bool = False
#     vae_up_block: bool = False
#     vae_down_block: bool = False

    ada_text_config: Union[None, dict] = None
        
        

class CNNAdapter(nn.Module):
    def __init__(self, hidden_size, emb_size, act='relu', init_std=1e-3):
        super().__init__()
        
        self.down_sample = nn.Conv2d(hidden_size, emb_size, 1, bias=False)
        self.act = ACT2FN[act]
        self.up_sample = nn.Conv2d(emb_size, hidden_size, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, init_std)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

    def forward(self, hidden_states):
        res = hidden_states
        hidden_states = self.down_sample(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.up_sample(hidden_states)
        hidden_states = hidden_states + res
        return hidden_states
    
class TransformerAdapter(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 emb_size, 
                 init_std=1.0e-3, 
                 norm_type='layer', 
                 act='relu',
                 preln=True,
                 useln=True):
        super().__init__()
        self.preln = preln
        self.useln = useln
        self.linear_down = nn.Linear(hidden_size, emb_size)
        self.act = ACT2FN[act]
        self.linear_up = nn.Linear(emb_size, hidden_size)
        if norm_type == 'layer' and useln:
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
        if self.useln:
            if self.preln:
                hidden_states = self.norm(hidden_states)
        hidden_states = self.linear_down(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_up(hidden_states)
        if self.useln:
            if not self.preln:
                hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states + res
        return hidden_states

        
