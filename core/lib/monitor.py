import pandas as pd
import tqdm
import os

class Monitor:
    def __init__(self, args):
        # save
        self.args = args

    def create_progress_bar(self, n_step):
        # initialize progress bar
        if progress_bar:
            self.bar = tqdm.tqdm(range(n_step))
        # initialize writer
        self.csv_data = {}
        self.global_step = 0

    def __update_time(self):
        self.bar.update(self.args.n_logging)

    def __update_description(self, **kwargs):
        _kwargs = {}
        for key in kwargs:
            for term in ['loss', 'length']:
                if term in key:
                    _kwargs[key] = f'{kwargs[key]:0.6f}'
        self.bar.set_postfix(**_kwargs)

    def __display(self):
        self.bar.display()

    def step(self, info):
        # extract stats from all stations
        # update progress bar
        self.__update_time()
        self.__update_description(**info)
        self.__display()
        # log to csv
        self.__update_csv(info)
        self.global_step += self.args.n_logging

    ####################################################################################
    # MODIFY HERE
    ####################################################################################
    @property
    def label(self):
        args = self.args
        label = f'{args.dataset}_{args.n_node}'
        return label

    def __update_csv(self, info):
        for key in info.keys():
            if key not in self.csv_data:
                self.csv_data[key] = [float(info[key])]
            else:
                self.csv_data[key].append(float(info[key]))

    def export_csv(self):
        # extract args
        args = self.args
        # save data to csv
        path = os.path.join(args.csv_dir, f'{self.label}.csv')
        df = pd.DataFrame(self.csv_data)
        df.to_csv(path, index=None)
