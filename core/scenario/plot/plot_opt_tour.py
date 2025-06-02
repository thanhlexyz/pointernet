import matplotlib.pyplot as plt
import networkx as nx
import lib
import os

def plot_instance(x, y):
    # create graph
    G = nx.DiGraph()
    # add nodes
    for u in range(len(x)):
        G.add_node(u, pos=x[u])
    # add optimal tour
    for u, v in zip(y[:-1], y[1:]):
        G.add_edge(u, v)
    G.add_edge(y[-1], y[0])
    # draw graph
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue')

def plot_opt_tour(args):
    # create dataset
    dataset = lib.create_dataset(args)
    dataset.prepare()
    # extract one example
    x, y = dataset.X[0], dataset.Y[0]
    plot_instance(x, y)
    # save figure
    path = os.path.join(args.figure_dir, f'{args.scenario}_{args.dataset}_{args.n_node}.pdf')
    plt.savefig(path)
    print(f'    - saved plot at {path=}')
