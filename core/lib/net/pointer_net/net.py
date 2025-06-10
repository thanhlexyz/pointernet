import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder

class Net(nn.Module):

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args
        # initialize model
        self.embed_layer = nn.Linear(args.n_input, args.n_embed)
        self.encoder     = nn.LSTM(input_size=args.n_embed,
                                   hidden_size=args.n_hidden,
                                   num_layers=1,
                                   batch_first=True)
        self.c0_dec      = nn.Parameter(torch.empty(args.n_embed))
        self.decoder     = nn.LSTM(input_size=args.n_embed,
                                   hidden_size=args.n_hidden,
                                   num_layers=1,
                                   batch_first=True)
        # self.attention = ...

    def forward(self, x_enc, x_dec):
        # x: batch_size, n_node, n_input
        e_enc = self.embed_layer(x_enc)
        e_dec = self.embed_layer(x_dec)
        # e: batch_size, n_node, n_embed
        enc, (hn_enc, cn_enc) = self.encoder(e)
        dec, _ = self.decoder(e_dec, (hn_enc, self.c0_dec))

        print(c0_dec
        print(enc.shape, hn_enc.shape, cn_enc.shape)
        exit()
