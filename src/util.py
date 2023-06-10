import os, pickle, hashlib
import collections

import numpy as np
import torch


class color:
    BOLD   = '\033[1m\033[48m'
    END    = '\033[0m'
    ORANGE = '\033[38;5;202m'
    BLACK  = '\033[38;5;240m'


def create_logger(args):
    from torch.utils.tensorboard import SummaryWriter
    """Use hyperparms to set a directory to output diagnostic files."""

    arg_dict = args.__dict__
    assert "seed" in arg_dict, \
    "You must provide a 'seed' key in your command line arguments"
    assert "logdir" in arg_dict, \
    "You must provide a 'logdir' key in your command line arguments."
    assert "env" in arg_dict, \
    "You must provide a 'env' key in your command line arguments."

    # sort the keys so the same hyperparameters will always have the same hash
    arg_dict = collections.OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

    # remove seed so it doesn't get hashed, store value for filename
    # same for logging directory
    run_name = arg_dict.pop('run_name')
    seed = str(arg_dict.pop("seed"))
    logdir = str(arg_dict.pop('logdir'))
    env_name = str(arg_dict['env'])

    # see if this run has a unique name, if so then that is going to be the name of the folder
    if run_name is not None:
        logdir = os.path.join(logdir, env_name)
        output_dir = os.path.join(logdir, run_name)
        # Check if policy name already exists. If it does, increment filename
        index = ''
        while os.path.exists(output_dir + index):
            if index:
                index = '_(' + str(int(index[2:-1]) + 1) + ')'
            else:
                index = '_(1)'
        output_dir += index
    else:
        # see if we are resuming a previous run, if we are mark as continued
        if hasattr(args, 'previous') and args.previous is not None:
            if args.exchange_reward is not None:
                output_dir = args.previous[0:-1] + "_NEW-" + args.reward
            else:
                print(args.previous[0:-1])
                output_dir = args.previous[0:-1] + '-cont'
        else:
            # get a unique hash for the hyperparameter settings, truncated at 10 chars
            arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6] + '-seed' + seed
            logdir     = os.path.join(logdir, env_name)
            output_dir = os.path.join(logdir, arg_hash)

    # create a directory with the hyperparm hash as its name, if it doesn't
    # already exist.
    os.makedirs(output_dir, exist_ok=True)

    # Create a file with all the hyperparam settings in human-readable plaintext,
    # also pickle file for resuming training easily
    info_path = os.path.join(output_dir, "experiment.info")
    pkl_path = os.path.join(output_dir, "experiment.pkl")
    with open(pkl_path, 'wb') as file:
        pickle.dump(args, file)
    with open(info_path, 'w') as file:
        for key, val in arg_dict.items():
            file.write("%s: %s" % (key, val))
            file.write('\n')

    logger = SummaryWriter(output_dir, flush_secs=0.1) # flush_secs=0.1 actually slows down quite a bit, even on parallelized set ups
    print("Logging to " + color.BOLD + color.ORANGE + str(output_dir) + color.END)

    logger.dir = output_dir
    return logger

# TODO: make this function less stupid.
def args_type(default):
    def parse_string(x):
        
        if default is None:
            return x
        if x == 'None':
            return None
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)

def recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            recursive_update(base[key], value)
        else:
            base[key] = value


def train_normalizer(env_fn, policy, min_timesteps, max_traj_len=1000, noise=0.5):
    with torch.no_grad():
        env = env_fn()
        env.dynamics_randomization = False

        total_t = 0
        while total_t < min_timesteps:
            state = env.reset()
            done = False
            timesteps = 0

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            while not done and timesteps < max_traj_len:
                if noise is None:
                    action = policy.forward(state, update_norm=True, deterministic=False).numpy()
                else:
                    action = policy.forward(state, update_norm=True).numpy() + np.random.normal(0, noise, size=policy.action_dim)
                state, _, done, _ = env.step(action)
                # env.render()
                timesteps += 1
                total_t += 1


def train_model_normalizer(env_fn, model, policy, min_timesteps, max_traj_len=1000, noise=0.5):
    with torch.no_grad():
        env = env_fn()
        env.dynamics_randomization = False

        total_t = 0
        while total_t < min_timesteps:
            state = env.reset()
            done = False
            timesteps = 0

            if hasattr(policy, 'init_hidden_state'):
                policy.init_hidden_state()

            while not done and timesteps < max_traj_len:
                if noise is None:
                    action = policy.forward(state, update_norm=False, deterministic=False).numpy()
                else:
                    action = policy.forward(state, update_norm=False).numpy() + np.random.normal(0, noise, size=policy.action_dim)
                model(state, action, update_norm=True) # Switch the update_norm to model.
                state, _, done, _ = env.step(action)
                # env.render()
                timesteps += 1
                total_t += 1