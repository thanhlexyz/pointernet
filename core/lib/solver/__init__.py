from . import pretrain, optimal, sampling, random

def create(args):
    return eval(args.solver).Solver(args)
