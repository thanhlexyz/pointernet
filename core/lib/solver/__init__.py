from . import pretrain

def create(args):
    return eval(args.solver).Solver(args)
