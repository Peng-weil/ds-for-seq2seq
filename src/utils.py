import argparse
import inspect
import math
import os
import pickle
import random
import re
import subprocess
import sys
import time

import numpy as np
import torch
from torch import optim

from .logger import create_logger

FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

CUDA = True


def get_sorting_result(x, sort_type):
    order_of_input = int(len(x)**0.5)
    matrix_of_input = np.array(x).reshape(order_of_input, order_of_input)
    sorting_result = []

    if sort_type == "SMR":
        return x
    elif sort_type == "SMC":
        sorting_result = matrix_of_input.T.reshape(1, -1).tolist()[0]
    elif sort_type == "SMD":
        for SMD_idx in range(order_of_input):
            for m in range(SMD_idx):
                sorting_result.append(matrix_of_input[m, SMD_idx])
            for n in range(SMD_idx, -1, -1):
                sorting_result.append(matrix_of_input[SMD_idx, n])
    elif sort_type == "counter-SMD":
        for tr_idx in range(order_of_input):
            for m in range(tr_idx):
                sorting_result.append(matrix_of_input[tr_idx, m])
            for n in range(tr_idx, -1, -1):
                sorting_result.append(matrix_of_input[n, tr_idx])
    return sorting_result


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def mkdir(path):
    """
    Create floder by path.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        return


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    if not CUDA:
        return args
    return [None if x is None else x.cuda() for x in args]


def init_exp(params):
    """
    Create a directory to store the experiment.
    """
    assert len(params.exp_name) > 0

    # dump parameters

    # create the sweep path if it does not exist
    sweep_path = os.path.join(params.dump_path, params.exp_name)
    mkdir(sweep_path)

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == '':
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        while True:
            exp_id = ''.join(random.choice(chars) for _ in range(10))
            if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                break
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    mkdir(params.dump_path)

    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'),
                             'wb'))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'),
                           rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger
