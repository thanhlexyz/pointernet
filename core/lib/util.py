import torch
from torch.nn.utils.rnn import PackedSequence

def get_tour_length(x: PackedSequence, y: PackedSequence) -> torch.Tensor:
    # Gather the coordinates based on the permutation in y
    # x: (bs, n_node, 2)
    # y: (bs, n_node)
    
    # Unpack PackedSequence objects
    x_unpacked, x_lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
    y_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(y, batch_first=True)
    x_lengths = x_lengths.to(x_unpacked.device)
    
    bs = x_unpacked.size(0)
    
    # step 1: for each batch i, pad x[i] with value of y[i][0]
    # Get the coordinate at position y[i][0] for each batch i
    batch_indices = torch.arange(bs, device=x_unpacked.device)
    first_tour_indices = y_unpacked[:, 0]  # y[i][0] for each batch i
    padding_coords = x_unpacked[batch_indices, first_tour_indices]  # (bs, coord_dim)
    
    # Create padded version where padding positions use the padding_coords
    x_padded = x_unpacked.clone()
    max_seq_len = x_unpacked.size(1)
    
    # Create mask for padding positions (beyond actual sequence length)
    padding_mask = torch.arange(max_seq_len, device=x_unpacked.device).unsqueeze(0) >= x_lengths.unsqueeze(1)
    
    # Apply padding: where padding_mask is True, use padding_coords
    x_padded[padding_mask] = padding_coords.unsqueeze(1).expand(-1, max_seq_len, -1)[padding_mask]
    
    permuted_x = x_padded[torch.arange(bs).unsqueeze(1), y_unpacked]
    # Compute the distance between consecutive nodes
    diffs = permuted_x[:, 1:] - permuted_x[:, :-1]
    distances = torch.norm(diffs, dim=-1)
    # Add the distance from the last node to the first to complete the tour
    closing_distance = torch.norm(permuted_x[:, 0] - permuted_x[:, -1], dim=-1)
    # Total tour length
    return distances.sum(dim=-1) + closing_distance
