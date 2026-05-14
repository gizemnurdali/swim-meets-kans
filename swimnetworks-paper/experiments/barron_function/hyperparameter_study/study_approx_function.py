import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from datetime import datetime

from time import time

import pandas as pd

from swimnetworks import Dense, Linear

def barron_e(x, n_dim):
    a_coeff = 2*np.arange(1, n_dim+1)/n_dim - 1
    a_coeff = a_coeff.reshape((1, -1))
    norm1 = np.linalg.norm(x-a_coeff, ord=2, axis=1)
    norm2 = np.linalg.norm(x+a_coeff, ord=2, axis=1)
    return np.sqrt(3/2) * (norm1 - norm2).reshape((-1, 1))

def singleneuron(x, n_dim):
    return np.clip(x[:, 0], 0, 1e10).reshape((-1, 1))

def l2_error_relative(f_approx, f_true):
    return np.linalg.norm(f_approx-f_true, ord=2) / np.linalg.norm(f_true, ord=2)

def setup_task(test_function, n_dim=2, xlim=[-3, 3], random_state = 1, n_points_train = 20000, n_points_test = 10000):
    """
    Setup the true_test_function as task to learn, on train and test data.
    """
    rng = np.random.default_rng(random_state)
    x_train = rng.uniform(low=xlim[0], high=xlim[1], size=(n_points_train, n_dim))
    x_test = rng.uniform(low=xlim[0], high=xlim[1], size=(n_points_test, n_dim))

    if test_function == 'BarronE':
        true_test_function = barron_e
    elif test_function == 'singleneuron':
        true_test_function = singleneuron
    else:
        raise ValueError(f"Test function {test_function} unknown.")

    f_train = true_test_function(x_train, n_dim=n_dim).reshape((-1, 1))
    f_test = true_test_function(x_test, n_dim=n_dim).reshape((-1, 1))
    return x_train, x_test, f_train, f_test

def train_sampling(x_train, f_train, n_layers, n_width, regularization_scale, random_seed=1):
    steps = []
    for k_layer in range(n_layers):
        steps.append((f"fcn{k_layer+1}", Dense(layer_width=n_width, activation=np.sin, parameter_sampler='tanh', random_seed=1 + random_seed + k_layer * 1234)))
    steps.append(("lin", Linear(regularization_scale=regularization_scale)))
    model = Pipeline(steps=steps, verbose=False)
    t0 = time()
    model.fit(x_train, f_train)
    t_fit = time()
    return lambda x: model.transform(x), model, t_fit-t0

def train_randomfeature(x_train, f_train, n_layers, n_width, regularization_scale, random_seed=1):
    steps = []
    for k_layer in range(n_layers):
        steps.append((f"fcn{k_layer+1}", Dense(layer_width=n_width, activation=np.sin, parameter_sampler='random', random_seed=1 + k_layer*1234 + random_seed)))
    steps.append(("lin", Linear(regularization_scale=regularization_scale)))
    model = Pipeline(steps=steps, verbose=False)
    t0 = time()
    model.fit(x_train, f_train)
    t_fit = time()
    return lambda x: model.transform(x), model, t_fit-t0

def train_model(model_type, n_layers, n_width, x_train, f_train, random_seed):
    if model_type=='sampling':
        return train_sampling(x_train=x_train, f_train=f_train, n_layers=n_layers, n_width=n_width, regularization_scale=1e-10, random_seed=random_seed)
    if model_type=='randomfeatures':
        return train_randomfeature(x_train=x_train, f_train=f_train, n_layers=n_layers, n_width=n_width, regularization_scale=1e-10, random_seed=random_seed)
    raise ValueError(f'model_type = {model_type} not supported.')

def evaluate_model(model, x_test, f_test):
    """
    Given a model as a lambda, evaluate the relative l2 error on x_test and f_test data.
    """
    f_test_approx = model(x_test)
    l2_relative = l2_error_relative(f_test_approx, f_test)
    return l2_relative

if __name__ == '__main__':
    xlim = [-1, 1]
    n_points_train = 10000
    n_points_test = 10000
    test_function = 'BarronE'

    experiment_data = []
    for n_dim in [1, 2, 3, 4, 5, 10]:
        x_train, x_test, f_train, f_test = setup_task(test_function=test_function, n_dim=n_dim, xlim=xlim, n_points_train=n_points_train, n_points_test=n_points_test)
        for n_layers in [1, 2, 3]:
            for n_width in (64*2**np.linspace(0, np.log(4096/64)/np.log(2), 10)).astype(int):
                for model_type in ['sampling']:
                    for random_seed in [1, 2, 3, 4, 5]:
                        model, _, time_fit = train_model(model_type, n_layers=n_layers, n_width=int(n_width), x_train=x_train, f_train=f_train, random_seed=random_seed)
                        l2_relative = evaluate_model(model, x_test, f_test)
                        experiment = {
                            'model_type': model_type,
                            'n_width': n_width,
                            'n_layers': n_layers,
                            'l2_relative': l2_relative,
                            'time_fit': time_fit,
                            'xlim_min': xlim[0],
                            'xlim_max': xlim[1],
                            'n_dim': n_dim,
                            'n_points_train': n_points_train,
                            'n_points_test': n_points_test,
                            'random_seed': int(random_seed)
                        }
                        print(experiment)
                        experiment_data.append(experiment)
    
    experiment_df = pd.DataFrame.from_dict(experiment_data)
    filename = f'experiment_{datetime.utcnow().strftime("%Y-%m-%d")}_{test_function}_{model_type}_with_sin_{n_points_train}pts_{np.random.randint(low=1000, high=9999999, size=1)[0]}'
    print(filename)
    filename = f'./experiments/barron_function/hyperparameter_study/{filename}.csv'
    experiment_df.to_csv(filename)
