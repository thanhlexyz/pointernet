from torch.nn.utils.rnn import PackedSequence
from beartype import beartype
import torch

@beartype
def get_tour_length(x_packed: PackedSequence, y_packed: PackedSequence) -> torch.Tensor:
    # x: (bs, n_node, 2)
    # y: (bs, n_node)
    # get start node
    x, N = torch.nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
    y, _ = torch.nn.utils.rnn.pad_packed_sequence(y_packed, batch_first=True)
    n_max = N.max()
    bs = x.shape[0]
    y0 = y[:, 0]
    x0 = x[torch.arange(bs), y0]
    for i in range(bs):
        n = N[i]
        x[i, n:, :] = x0[i].unsqueeze(0).repeat(n_max - n, 1)
        y[i, n:] = y0[i].unsqueeze(0).repeat(n_max - n)
    # gather the coordinates based on the permutation in y
    permuted_x = x[torch.arange(x.shape[0]).unsqueeze(1), y]
    # Compute the distance between consecutive nodes
    diffs = permuted_x[:, 1:] - permuted_x[:, :-1]
    distances = torch.norm(diffs, dim=-1)
    # Add the distance from the last node to the first to complete the tour
    closing_distance = torch.norm(permuted_x[:, 0] - permuted_x[:, -1], dim=-1)
    # Total tour length
    return distances.sum(dim=-1) + closing_distance
