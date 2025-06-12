import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
import numpy as np
import os

def load(args):
    label = f'{args.mode}/{args.dataset}_{args.n_node}_{args.solver}'
    path = os.path.join(args.csv_dir, f'{label}.csv')
    df = pd.read_csv(path)
    y = df[args.metric].to_numpy()
    return float(y[0])

def plot_bar(args):
    plt.cla()
    plt.clf()
    # scenarios
    args.dataset = 'tsp'
    args.mode = 'test'
    args.n_node = 5
    solvers = ['random', 'optimal', 'pretrain', 'sampling']
    labels, values = [], []
    for solver,  in it.product(solvers):
        # assign args
        args.solver = solver
        label = f'{args.solver}'
        # load csv
        try:
            y = load(args)
        except:
            raise
        else:
            labels.append(label)
            values.append(y)
    # plt
    print(labels, values)
    plt.bar(labels, values)
    # decorate
    plt.xlabel('solver')
    plt.ylabel(f'{args.metric}')
    plt.tight_layout()
    # save figure
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.metric}_{args.n_node}.jpg')
    plt.savefig(path)
