import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
import numpy as np
import os

def smooth_line(x, y, args):
    x = x.reshape(-1, args.n_smooth)
    x = x[:, 0]
    y = y.reshape(-1, args.n_smooth)
    y = np.mean(y, axis=1)
    return x, y

def load(args):
    label = f'{args.dataset}_{args.n_node}'
    path = os.path.join(args.csv_dir, f'{label}.csv')
    df = pd.read_csv(path)
    x = df['epoch'].to_numpy()
    y = df[args.metric].to_numpy()
    x, y = smooth_line(x, y, args)
    return x, y

def plot_line(args):
    # scenarios
    datasets = ['tsp']
    n_nodes = [10]
    for dataset, n_node in it.product(datasets, n_nodes):
        # assign args
        args.dataset = dataset
        args.n_node = n_node
        label = f'{args.dataset}_{args.n_node}'
        # load csv
        try:
            x, y = load(args)
            # plot
            plt.plot(x, y, '-o', label=label)
        except:
            raise
    # decorate
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(f'{args.metric}')
    plt.yscale('log')
    # save figure
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.metric}_{label}.jpg')
    plt.savefig(path)
