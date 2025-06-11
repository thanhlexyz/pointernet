import torch
import lib

def debug(args):
    # create sample data
    x = torch.rand(args.batch_size, args.n_node, args.n_input)
    # create net
    net = lib.create_net(args)
    # test forward
    p, y = net(x)
