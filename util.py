import numpy as np
import inspect
import random
import os
import torch

def check_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def arg_parser():
    """
    Create an empty argparse.ArgumentParser.
    """
    import argparse
    return argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def get_objects(name_space):
    res = {}
    for name, obj in inspect.getmembers(name_space):
        if inspect.isclass(obj):
            res[name] = obj
    return res

def set_global_seeds(i):
    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i) # cpu
    torch.cuda.manual_seed(i)  # gpu
    torch.backends.cudnn.deterministic=True # cudnn

def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator / denominator
    return softmax

def path_join(a,b):
    return os.path.join(a,b)

def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    else:
        return tensor

save4float = lambda x:str(round(x,4))
