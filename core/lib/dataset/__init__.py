from torch.utils.data import DataLoader
from . import tsp
import torch
import os

def collate_fn(items):
    # TODO: sort bằng cơm rồi tự pack thì efficient hơn
    x = torch.nn.utils.rnn.pack_sequence([_['x'] for _ in items], enforce_sorted=False)
    y = torch.nn.utils.rnn.pack_sequence([_['y'] for _ in items], enforce_sorted=False)
    return x, y

def create(args):
    dataloader_dict = {}
    mode = args.mode
    dataset = eval(args.dataset).Dataset(mode, args)
    dataset.prepare()
    dataloader_dict[mode] = DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                # num_workers=os.cpu_count())
                                num_workers=1)
    return dataloader_dict
