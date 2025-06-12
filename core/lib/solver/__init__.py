from . import pretrain, optimal, sampling, random, active_search

def create(args):
    return eval(args.solver).Solver(args)
