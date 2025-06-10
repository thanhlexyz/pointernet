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
            # outs: (batch, n_node)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices using beam search
            batch_size = masked_outs.size(0)
            vocab_size = masked_outs.size(1)
            beam_width = self.args.beam_width
            
            # For the first step, select top beam_width candidates
            if len(outputs) == 0:
                # Get top-k scores and indices using log probabilities for numerical stability
                log_probs = masked_outs.log()
                beam_log_probs, indices = log_probs.topk(beam_width, dim=1)
                # beam_log_probs: (batch_size, beam_width)
                # indices: (batch_size, beam_width)
                
                # Initialize beam scores and sequences
                beam_scores = beam_log_probs  # (batch_size, beam_width)
                beam_sequences = indices.unsqueeze(-1)  # (batch_size, beam_width, 1)
                
                # Take the first beam for now (we'll expand in next steps)
                indices = indices[:, 0]  # (batch_size,)
                
            else:
                # For subsequent steps
                # Calculate scores for each beam using log probabilities
                step_log_probs = masked_outs.log()  # (batch_size, vocab_size)
                
                # Add current log probs to cumulative log probs (multiplication becomes addition in log space)
                # Apply length normalization: (l+1)^alpha / 6^alpha, where l is the current sequence length
                # This helps balance between sequence length and probability
                length_penalty = ((len(outputs) + 1) ** self.args.beam_alpha) / (6 ** self.args.beam_alpha)
                
                # For each beam's prediction, calculate cumulative score with each possible next token
                cumulative_log_probs = (beam_scores[:, 0].unsqueeze(-1) + step_log_probs) / length_penalty  # (batch_size, vocab_size)
                
                # Select top-k scores and indices
                beam_scores, word_indices = cumulative_log_probs.topk(beam_width, dim=1)  # (batch_size, beam_width)
                
                # Update beam sequences by appending new predictions
                # Note: Currently tracking only the top beam's history, but maintaining scores for all beams
                beam_sequences = torch.cat([
                    beam_sequences[:, 0].unsqueeze(1).expand(-1, beam_width, -1),
                    word_indices.unsqueeze(-1)
                ], dim=-1)
                
                # Take the first beam for now
                indices = word_indices[:, 0]  # (batch_size,)
            
            # Create one-hot pointers and update mask
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()
            mask = mask * (1 - one_hot_pointers)
            
            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, args.n_embed).byte()
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, args.n_embed)
            
            # Store outputs and pointers
            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden
