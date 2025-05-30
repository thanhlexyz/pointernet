import lib

def prepare(args):
    dataset = lib.create_dataset(args)
    dataset.prepare()
