import lib

def main(args):
    # create solver

    # create dataset
    dataset = lib.create_dataset(args)
    dataset.prepare()
    # create net
    net = lib.create_net(args)
