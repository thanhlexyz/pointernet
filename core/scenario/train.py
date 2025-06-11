import lib

def train(args):
    # create solver
    solver = lib.create_solver(args)
    # run training
    solver.train()
