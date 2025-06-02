import torch

def get_tour_length(x, y):
    # Gather the coordinates based on the permutation in y
    permuted_x = x[torch.arange(x.shape[0]).unsqueeze(1), y]
    # Compute the distance between consecutive nodes
    diffs = permuted_x[:, 1:] - permuted_x[:, :-1]
    distances = torch.norm(diffs, dim=-1)
    # Add the distance from the last node to the first to complete the tour
    closing_distance = torch.norm(permuted_x[:, 0] - permuted_x[:, -1], dim=-1)
    # Total tour length
    return distances.sum(dim=-1) + closing_distance
