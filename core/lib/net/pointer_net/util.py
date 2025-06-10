import torch.nn as nn
import typing as tp
import numpy as np
import torch

from torch.nn.utils.rnn import PackedSequence

def prepend(sequences: PackedSequence, tensor: torch.Tensor) -> PackedSequence:
    """Prepends a tensor to each sequence"""
    padded, lens = nn.utils.rnn.pad_packed_sequence(sequences)
    # repeat tensor batch_size times
    # tensor shape should be the same shape as each token in a sequence
    batch_size = padded.shape[1]
    padded = torch.cat(
        [tensor.repeat(1, batch_size, *[1] * (padded.ndim - 2)), padded], dim=0
    )
    return nn.utils.rnn.pack_padded_sequence(
        padded, lengths=lens + 1, enforce_sorted=False
    )

def unravel_index(
    indices: torch.Tensor, shape: tp.Tuple[int, ...]
) -> tp.Tuple[torch.Tensor, ...]:
    # Supper innefficient to copy to cpu and then back to cuda if indices
    # is a cuda tensor, but for now it suffices.
    device = indices.device
    unraveled_coords = np.unravel_index(indices.cpu().numpy(), shape)
    return tuple(torch.tensor(arr, device=device) for arr in unraveled_coords)
