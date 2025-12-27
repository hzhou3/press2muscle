# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import math
import logging
import torch
from einops import rearrange
from torch import nn
from .blocks.mingpt import Block
from .blocks.convolutions import Masked_conv, Masked_up_conv
from .pressure_net import MultiFoot_CBAM
from dataset.dataset_details import BIO_COLs

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    def __init__(self, dim, type='sine_frozen', max_len=1024, *args, **kwargs):
        super(PositionalEncoding, self).__init__()
        if 'sine' in type:
            rest = dim % 2
            pe = torch.zeros(max_len, dim + rest)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim + rest, 2).float() * (-math.log(10000.0) / (dim + rest)))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe[:, :dim]
            pe = pe.unsqueeze(0)  # [1,t,d]
            if 'ft' in type:
                self.pe = nn.Parameter(pe)
            elif 'frozen' in type:
                self.register_buffer('pe', pe)
            else:
                raise NameError
        elif type == 'learned':
            self.pe = nn.Parameter(torch.randn(1, max_len, dim))
        elif type == 'none':
            # no positional encoding
            pe = torch.zeros((1, max_len, dim))  # [1,t,d]
            self.register_buffer('pe', pe)
        else:
            raise NameError

    def forward(self, x, start=0):
        x = x + self.pe[:, start:(start + x.size(1))]
        return x


class Encoder_Config:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class Stack(nn.Module):
    """ A stack of transformer blocks.
        Used to implement a U-net structure """

    def __init__(self, block_size, n_layer=12, n_head=8, n_embd=256,
                 dropout=0.1, causal=False, down=1, up=1,
                 pos_type='sine_frozen', sample_method='conv',
                 pos_all=False):
        super().__init__()
        config = Encoder_Config(block_size, n_embd=n_embd, n_layer=n_layer, n_head=n_head, dropout=dropout,
                                causal=causal)
        self.drop = nn.Dropout(dropout)
        assert down == 1 or up == 1, "Unexpected combination"
        assert down in [1, 2] and up in [1, 2], "Not implemented"
        assert sample_method in ['cat', 'conv'], "Unknown sampling method"
        cat_down, slice_up = (down, up) if sample_method == 'cat' else (1, 1)
        self.cat_down, self.slice_up = cat_down, slice_up
        self.pos_all = pos_all
        self.blocks = nn.ModuleList([])
        self.pos = nn.ModuleList([])
        for i in range(config.n_layer):
            # Inside Block, standard transformer stuff happens.
            self.blocks.append(Block(config,
                                     in_factor=cat_down if i == 0 and cat_down > 1 else None,
                                     out_factor=slice_up if i == config.n_layer - 1 and slice_up > 1 else None))
            in_dim = config.n_embd * (cat_down if i == 0 and cat_down > 1 else 1)
            if pos_all or i == 0:
                self.pos.append(PositionalEncoding(dim=in_dim, max_len=block_size, type=pos_type))
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        self.down_conv, self.up_conv = None, None
        if sample_method == 'conv':
            if down == 2:
                self.down_conv = Masked_conv(config.n_embd, config.n_embd, pool_size=down, pool_type='max')
            elif up == 2:
                self.up_conv = Masked_up_conv(config.n_embd, config.n_embd)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x=None):
        t = x.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.cat_down > 1:
            if self.cat_down != 2:
                raise NotImplementedError
            else:
                x = rearrange(x, 'b (t t2) c -> b t (t2 c)', t2=2)

        if self.down_conv is not None:
            x = self.down_conv(x)

        x = self.drop(x)
        for i in range(len(self.blocks)):
            x = self.pos[i](x) if (i == 0 or self.pos_all) else x
            x = self.blocks[i](x, in_residual=not (i == 0 and self.cat_down > 1),
                               out_residual=not (i == (len(self.blocks) - 1) and self.slice_up > 1))
        if self.slice_up > 1:
            x = rearrange(x, 'b t (t2 c) -> b (t t2) c', t2=2)

        if self.up_conv is not None:
            x = self.up_conv(x)

        x = self.ln_f(x)  # (bs, seq/2, 384)
        logits = self.head(x)  # (bs, seq/2, 384)
        return logits


class TransformerAutoEncoder(nn.Module):
    """
    Model composed of an encoder and a decoder.
    """
    def __init__(self, in_dim=36, out_dim=8, n_layers=[4, 4], hid_dim=256, heads=4, dropout=0.0, e_dim=384, block_size=2048,
                 pos_type='sine_frozen', pos_all=False, sample_method='conv', embed_type='linear'):
        super().__init__()

        if not isinstance(hid_dim, list):
            hid_dim = [hid_dim]
        if len(hid_dim) != 1:
            raise NotImplementedError("Does not handle per-layer channel specification.")

        # Constrols masking of  attention in encoder / decoder.
        n_embd = hid_dim[0]
        self.in_dim = in_dim
        self.out_dim = out_dim

        if embed_type == 'cbam':
            self.emb = MultiFoot_CBAM(in_channels=in_dim, n_embd=n_embd)
        else:
            raise NotImplementedError("Does not handle per-layer channel specification.") 
        self.bio_embed = nn.Sequential(
            nn.Linear(len(BIO_COLs), n_embd),
            nn.ReLU(),  
            nn.Linear(n_embd, n_embd),
        )

        self.uid_emb_beta = nn.Sequential(
            nn.Linear(len(BIO_COLs), n_embd),
            nn.ReLU(),  
            nn.Linear(n_embd, n_embd),
        )
        self.uid_emb_sigma = nn.Sequential(
            nn.Linear(len(BIO_COLs), n_embd),
            nn.ReLU(),  
            nn.Linear(n_embd, n_embd),
        )

        self.encoder_stacks = nn.ModuleList(
            [Stack(block_size=block_size, n_layer=n_layers[0], n_head=heads, n_embd=n_embd,
                   dropout=dropout, down=1, pos_type=pos_type,
                   pos_all=pos_all, sample_method=sample_method)])

        
        self.emb_in, self.emb_out = n_embd, e_dim
        self.encoder_dim = e_dim

        dim = n_embd

        self.cls = nn.Sequential(
                nn.Linear(dim, dim),  
                nn.ReLU(),
                nn.Linear(dim, 25)  
            )

        # Shared base layers
        self.shared_layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            
        )

        # regress heads
        self.heads = nn.ModuleList([
            nn.Linear(dim, 1) for i in range(self.out_dim)
        ])


    def encoder(self, x):
        """ Calls each encoder stack sequentially """
        o = self.encoder_stacks[0](x)
        return o

    
    def personalize(self, x, uid):
        """ feature-wise scale by uid """
        beta = self.uid_emb_beta(uid)
        gamma = self.uid_emb_sigma(uid)
        x = x * gamma.unsqueeze(1) + beta.unsqueeze(1)

        return x

    def forward(self, x, uid):
        """"
        Run the forward pass of the encoder
        """

        ''' linear embed or MultiFoot'''
        x_emb = self.emb(x)  # (bs, seq, 512)

        # ''' personalization '''
        x_emb = self.personalize(x_emb, uid)

        ''' bio token '''
        bio_emd = self.bio_embed(uid).unsqueeze(1)
        x_emb = torch.cat([bio_emd, x_emb], dim=1)

        ''' encoder '''   
        hid = self.encoder(x=x_emb)  
        hid = hid[:, 1:]

        ''' regression  and classification'''
        shared_features = self.shared_layers(hid)  # Apply shared layers

        ''' regress heads'''
        y_pred = torch.cat([head(shared_features) for head in self.heads], dim=-1)

        return y_pred, None 