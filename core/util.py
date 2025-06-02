import argparse
import torch
import os

torch.set_printoptions(4, sci_mode=False)

def create_folders(args):
    ls = [args.csv_dir, args.figure_dir, args.dataset_dir, args.model_dir]
    for folder in ls:
        if not os.path.exists(folder):
            os.makedirs(folder)

def set_default_device(args):
    torch.set_default_device(args.device)

base_folder = os.path.dirname(os.path.dirname(__file__))

def get_args():
    # create args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, default='main')
    parser.add_argument('--mode', type=str, default='test')
    # prepare
    parser.add_argument('--n_train_instance', type=int, default=70000)
    parser.add_argument('--n_test_instance', type=int, default=30000)
    parser.add_argument('--n_node', type=int, default=10)
    # solver
    parser.add_argument('--dataset', type=str, default='tsp')
    parser.add_argument('--n_train_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    # dnn hyperparameter
    parser.add_argument('--net', type=str, default='pointer_net')
    parser.add_argument('--n_embed', type=int, default=16)
    parser.add_argument('--n_hidden', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--bidirectional', action='store_true')
    # data directory
    parser.add_argument('--dataset_dir', type=str, default='../data/dataset')
    parser.add_argument('--figure_dir', type=str, default='../data/figure')
    parser.add_argument('--model_dir', type=str, default='../data/model')
    parser.add_argument('--csv_dir', type=str, default='../data/csv')
    # plot
    parser.add_argument('--metric', type=str, default='reward')
    parser.add_argument('--n_smooth', type=int, default=1)
    # other
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda:0')
    else:
        parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--score', type=str, default='actor')
    parser.add_argument('--load_state_dict', action='store_true')
    parser.add_argument('--clear_buffer', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    # parse args
    args = parser.parse_args()
    # create folders
    create_folders(args)
    # set default device cuda
    # set_default_device(args)
    # additional args
    args.n_instance = eval(f'args.n_{args.mode}_instance')
    return args
