import lib

def main(args):
    # create solver
    solver = lib.create_solver(args)
    # run training
    solver.train()
