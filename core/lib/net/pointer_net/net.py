import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder

class Net(nn.Module):
    # Pointer-Net

    def __init__(self, args):
        super().__init__()
        # save args
        self.args = args
        self.embedding = nn.Linear(2, args.n_embed)
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.decoder_input0 = Parameter(torch.FloatTensor(args.n_embed), requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, x):
        # :param Tensor x: Input sequence
        # :return: Pointers probabilities and indices

        # extract args
        args = self.args
        batch_size = x.size(0)
        input_length = x.size(1)

        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        x = x.view(batch_size * input_length, -1)
        embedded_inputs = self.embedding(x).view(batch_size, input_length, -1)

        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs,
                                                       encoder_hidden0)
        if args.bidirectional:
            decoder_hidden0 = (torch.cat(encoder_hidden[0][-2:], dim=-1),
                               torch.cat(encoder_hidden[1][-2:], dim=-1))
        else:
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs,
                                                           decoder_input0,
                                                           decoder_hidden0,
                                                           encoder_outputs)

        return outputs, pointers
