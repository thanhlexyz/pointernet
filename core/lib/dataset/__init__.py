from torch.utils.data import DataLoader
from . import tsp
import os

def create(args):
    dataloader_dict = {}
    for mode in ['train', 'val', 'test']:
        dataset = eval(args.dataset).Dataset(mode, args)
        dataset.prepare()
        dataloader_dict[mode] = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True if mode == 'train' else False,
                                    num_workers=os.cpu_count())
    return dataloader_dict
