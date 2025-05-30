from torch.utils.data import Dataset
from joblib import Parallel, delayed
import numpy as np
import torch
import time
import tqdm
import os

from .util import solve_optimal_tsp

class Dataset(Dataset):

    def __init__(self, args):
        # save args
        self.args = args

    def prepare(self):
        # extract args
        args = self.args
        # check if data exists
        label = f'{args.dataset}_{args.n_node}_{args.n_instance}.npz'
        path = os.path.join(args.dataset_dir, label)
        if os.path.exists(path):
            data   = np.load(path)
            self.X = data['X']
            self.Y = data['Y']
            print(f'    - loaded {path=}')
        else:
            # generate 2d coordinates for inputs
            tic = time.time()
            self.X = np.random.rand(args.n_instance, args.n_node, 2)
            self.Y = Parallel(n_jobs=os.cpu_count())\
                             (delayed(solve_optimal_tsp)(self.X[i, :]) \
                                  for i in tqdm.tqdm(range(args.n_instance)))
            self.Y = np.array(self.Y)
            # save data
            label = f'{args.dataset}_{args.n_node}_{args.n_instance}.npz'
            path = os.path.join(args.dataset_dir, label)
            np.savez_compressed(path, X=self.X, Y=self.Y)
            toc = time.time()
            dt  = toc - tic
            print(f'    - saved {path=} {dt=:0.1f} (s)')
