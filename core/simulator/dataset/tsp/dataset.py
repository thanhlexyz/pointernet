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
        n_instance = eval(f'args.n_{self.mode}_episode')
        label = f'{args.dataset}_{self.mode}_{args.n_node}_{n_instance}.npz'
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
            self.X = np.random.rand(n_instance, args.n_node, 2)
            self.Y = Parallel(n_jobs=os.cpu_count())\
                             (delayed(solve_optimal_tsp)(self.X[i, :]) \
                                  for i in tqdm.tqdm(range(n_instance)))
            self.Y = np.array(self.Y)
            # save data
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
