from .dataset import create_dataset
from .monitor import Monitor
from .solver import Solver
from . import pointer_net

def create_solver(args):
    return Solver(args)
