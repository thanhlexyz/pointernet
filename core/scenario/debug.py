import torch
import lib

def debug(args):
    # create sample data
    dataloaders = lib.dataset.create(args)
    dataloader  = dataloaders['test']
    # create net
    actor = lib.pointer_net.Actor(args)
    critic = lib.pointer_net.Critic(args)
    # training loop
    for x, y in dataloader:
        # extract data
        x = x.to(args.device)
        # get actor prediction
        log_likelihood, y_hat = actor(x)
        print(f'{y_hat=}')
        l = critic(x)
        print(f'{l=}')
        exit()
        
