import simulator as sim

def main(args):
    # create solver
    solver = sim.solver.create(args)
    # run training
    if args.mode == 'train':
        solver.train()
    elif args.mode == 'test':
        solver.test()
    else:
        raise NotImplementedError
