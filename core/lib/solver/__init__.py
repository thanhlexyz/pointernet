from . import pretrain, optimal

def create(args):
    return eval(args.solver).Solver(args)
