import lib

def main(args):
    # create solver
    solver = lib.solver.create(args)
    # run training
    solver.eval(args.mode)()
