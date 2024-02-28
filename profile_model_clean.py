import os
import torch
import argparse

from src.model.factory import initialize_model


def load_model(tensorboard_path: str):
    """
    Args:
    tensorboard_path: the path of the directory of tensorboard
    """
    arg_file_path = os.path.join(tensorboard_path, "args.txt")
    model_config = reload_argparse(arg_file_path)

    net = initialize_model(model_config)

    model_pth_path = os.path.join(tensorboard_path, "best_model.pth")
    saved_state_dict = torch.load(model_pth_path, map_location=torch.device('cpu'))

    net.load_state_dict(saved_state_dict)
    print("successfully load model")
    return net, model_config


def reload_argparse(file_path: str):
    d = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, value = line.strip('\n').split(',')
            # print(f"{key}, {value}\n")
            try:
                re = eval(value)
            except:
                re = value
            d[key] = re

    return argparse.Namespace(**d)


parser = argparse.ArgumentParser(description='predict FLOPS')
parser.add_argument('path', type=str,
                    help="directory to model file")
parser.add_argument('--alpha', type=float,
                    default=None,
                    help="if set the value of alpha of sparsemax")
parser.add_argument('--workload', type=str,
                    default="random")
