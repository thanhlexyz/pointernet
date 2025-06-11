import torch.nn as nn
import torch

class Greedy(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, log_p):
		return torch.argmax(log_p, dim=1).long()
