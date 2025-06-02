import lib

def main(args):
    # create solver
    solver = lib.create_solver(args)
    # run mode
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    else:
        raise NotImplementedError
