import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from .attention import Attention

class Decoder(nn.Module):

    def __init__(self, args):
        # :param int embedding_dim: Number of embeddings in Pointer-Net
        # :param int hidden_dim: Number of hidden units for the decoder's RNN
        super(Decoder, self).__init__()
        self.args          = args
        self.input_to_hidden = nn.Linear(args.n_embed, 4 * args.n_hidden)
        self.hidden_to_hidden = nn.Linear(args.n_hidden, 4 * args.n_hidden)
        self.hidden_out = nn.Linear(args.n_hidden * 2, args.n_hidden)
        self.att = Attention(args)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        # :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        # :param Tensor decoder_input: First decoder's input
        # :param Tensor hidden: First decoder's hidden states
        # :param Tensor context: Encoder's outputs
        # :return: (Output probabilities, Pointers indices), last hidden state
        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)
        args = self.args
        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())
        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            # :param Tensor x: Input at time t
            # :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            # :return: Hidden states at time t (h, c), Attention probabilities (Alpha)

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            input = F.sigmoid(input)
            forget = F.sigmoid(forget)
            cell = F.tanh(cell)
            out = F.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * F.tanh(c_t)

            # Attention section
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = F.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        for _ in range(input_length):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            # Update mask to ignore seen indices
            mask  = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, args.n_embed).byte()
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, args.n_embed)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden
