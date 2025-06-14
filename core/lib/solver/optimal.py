import math

import lib.util as util
import lib

from .base import SolverBase

class Solver(SolverBase):

    def __init__(self, args):
        # save args
        self.args = args
        # create dataloader dict
        self.dataloader_dict = lib.dataset.create(args)
        # load monitor
        self.monitor = lib.Monitor(args)
        # n_test
        self.n_test = math.ceil(args.n_test_instance / args.batch_size)

    def test_epoch(self):
        # extract args
        args = self.args
        dataloader = self.dataloader_dict['test']
        # training loop
        for x, y in dataloader:
            # extract data
            x = x.to(args.device)
            y = y.to(args.device)
            l = util.get_tour_length(x, y)
            # gather info
            yield l.detach().cpu().numpy().tolist()
