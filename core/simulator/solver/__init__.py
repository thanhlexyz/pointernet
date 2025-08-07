from . import random, greedy, optimal

def create(args):
    return eval(args.solver).Solver(args)
