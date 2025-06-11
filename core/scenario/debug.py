import torch
import lib

def debug(args):
    # create sample data
    x = torch.rand(args.batch_size, args.n_node, args.n_input)
    # create net
    actor = lib.pointer_net.Actor(args)
    critic = lib.pointer_net.Critic(args)
    # test forward
    ll, y = actor(x)
    print(ll.shape, y.shape)
    v = critic(x)
    print(v.shape)
