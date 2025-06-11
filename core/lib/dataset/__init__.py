from . import tsp

def create_dataloader_dict(args):
    dataloader_dict = {}
    for mode in ['train', 'val', 'test']:
        dataset = eval(args.dataset)(mode, args)
        dataset.prepare()
        dataloader_dict[mode] = DataLoader(dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True of mode == 'train' else False,
                                    num_workers=os.cpu_count())
    return dataloader_dict
