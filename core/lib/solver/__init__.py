from . import pretrain, optimal, sampling

def create(args):
    return eval(args.solver).Solver(args)
