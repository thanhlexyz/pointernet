import lib

def main(args):
    # create solver
    solver = lib.solver.create(args)
    # run training
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    else:
        raise NotImplementedError
