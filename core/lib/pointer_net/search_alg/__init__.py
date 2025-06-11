from .greedy import Greedy

def create(args):
    if args.search_alg == 'greedy':
        return Greedy(args)
    else:
        raise NotImplementedError
