from . import tsp

def create_dataset(args):
    if args.dataset == 'tsp':
        return tsp.Dataset(args)
