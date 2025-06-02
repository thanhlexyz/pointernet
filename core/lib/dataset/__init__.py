from . import tsp

def create_dataset(mode, args):
    return eval(args.dataset).Dataset(mode, args)
