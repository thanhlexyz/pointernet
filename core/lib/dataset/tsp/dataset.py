from torch.utils.data import Dataset
from joblib import Parallel, delayed
import numpy as np
import torch
import time
import tqdm
import os

from .util import solve_optimal_tsp

class Dataset(Dataset):

    def __init__(self, mode, args):
        # save args
        self.args = args
        self.mode = mode

    def prepare(self):
        # extract args
        args = self.args
        # check if data exists
        n_instance = eval(f'args.n_{self.mode}_instance')
        label = f'{args.dataset}_{self.mode}_{args.n_node_min}_{args.n_node_max}_{n_instance}.npz'
        path = os.path.join(args.dataset_dir, label)
        if os.path.exists(path):
            data   = np.load(path)
            self.X = data['X']
            self.Y = data['Y']
            if args.verbose:
                print(f'    - loaded {path=}')
        else:
            # generate 2d coordinates for inputs
            print(f'[+] preparing {label}')
            tic = time.time()
            # need to rand from n_node_min to n_node_max, use packed sequence from torch.nn.utils.rnn
            n_nodes = np.random.randint(args.n_node_min, args.n_node_max + 1, size=(n_instance,))
            self.X = torch.nn.utils.rnn.pack_sequence(
                [torch.rand(n_nodes[i], 2) for i in range(n_instance)], 
                enforce_sorted=False,
            )
            # print(self.X)
            for i, x in enumerate(self.X):
                print(i, x.shape, torch.rand(n_nodes[i], 2).shape)
            exit()
            # use delayed parallel to solve optimal tsp
            self.Y = torch.nn.utils.rnn.pack_sequence(
                Parallel(n_jobs=os.cpu_count())(
                    delayed(solve_optimal_tsp)(self.X[i, :n_nodes[i]]) 
                        for i in tqdm.tqdm(range(n_instance))
                )
            )
            
            np.savez_compressed(path, X=self.X, Y=self.Y)
            toc = time.time()
            dt  = toc - tic
            print(f'    - saved {path=} {dt=:0.1f} (s)')
        # convert from double to float
        self.X = self.X.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        sample = {'x': self.X[i], 'y': self.Y[i]}
        return sample


if __name__ == '__main__':
    from util import get_args
    args = get_args()
    dataset = Dataset('train', args)
    dataset.prepare()