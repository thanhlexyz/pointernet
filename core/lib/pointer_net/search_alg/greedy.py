import torch.nn as nn
import torch

class Greedy(nn.Module):
	def __init__(self, args):
		super().__init__()

	def forward(self, log_prob):
		return torch.argmax(log_prob, dim=1).long()
