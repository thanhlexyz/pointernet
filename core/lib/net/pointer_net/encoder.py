import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, args):
        # :param Tensor embedding_dim: Number of embbeding channels
        # :param int hidden_dim: Number of hidden units for the LSTM
        # :param int n_layers: Number of layers for LSTMs
        # :param float dropout: Float between 0-1
        # :param bool bidir: Bidirectional
        super(Encoder, self).__init__()
        self.args = args
        self.n_hidden      = args.hidden_dim // 2 if args.bidirectional else args.n_hidden
        self.n_layer       = args.n_layer * 2 if args.bidirectional else args.n_layer
        self.lstm          = nn.LSTM(args.n_embed,
                                     self.n_hidden,
                                     self.n_layer,
                                     dropout=args.dropout,
                                     bidirectional=args.bidirectional)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                hidden):
        # :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        # :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        # :return: LSTMs outputs and hidden units (h, c)
        embedded_inputs = embedded_inputs.permute(1, 0, 2)
        outputs, hidden = self.lstm(embedded_inputs, hidden)
        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        # :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        # :return: Initiated hidden units for the LSTMs (h, c)
        batch_size = embedded_inputs.size(0)
        args = self.args
        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(args.n_layer,
                                                      batch_size,
                                                      args.n_hidden)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(args.n_layer,
                                                      batch_size,
                                                      args.n_hidden)
        return h0, c0
