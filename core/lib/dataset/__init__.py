from . import tsp

def create_dataset(args):
    return eval(args.dataset).Dataset(args)
