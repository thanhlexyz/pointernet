from . import pointer_net

def create_net(args):
    return eval(args.net).Net(args)
