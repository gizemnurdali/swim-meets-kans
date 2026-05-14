import numpy as np
from swimnetworks import Dense, Linear
from utils import swim_to_keras_model

from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from datetime import datetime
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Input

from time import time
import json
from sklearn.base import BaseEstimator
import pandas as pd

import scipy.sparse

from openml import datasets, tasks, runs, study

import sklearn.compose as compose
import sklearn.preprocessing as preprocessing
import sklearn.impute as impute

import warnings


class TensorflowTransformer(BaseEstimator):
    def __init__(self, tensorflow_model, encoding_map=None, decoding_map=None, batch_size=1024, epochs=100, verbose=0, callbacks=None) -> None:
        self.tensorflow_model = tensorflow_model
        self.encoding_map = encoding_map
        self.decoding_map = decoding_map
        self.historian = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.callbacks = [] if callbacks is None else callbacks

    def _clean_inputs(self, X, y):
        if y is not None:
            if isinstance(y, (pd.DataFrame, pd.Series)):
                y = y.to_numpy()
            if y.ndim == 1:
                y = y.reshape((-1, 1))
        return X, y

    def fit(self, X, y=None):
        X, y = self._clean_inputs(X, y)

        if self.encoding_map is not None:
            y, _ = self.encoding_map(y)
        self.historian = self.tensorflow_model.fit(X, y, validation_split=0, batch_size=self.batch_size, verbose=self.verbose, epochs=self.epochs, callbacks=self.callbacks)
        return self
    
    def predict(self, X):
        return self.transform(X)

    def transform(self, X, y=None):
        _result = self.tensorflow_model(X).numpy()
        if self.decoding_map is not None:
            return self.decoding_map(_result)
        else:
            return _result

    # noinspection PyUnusedLocal
    def score(self, x, y, sample_weight=None):
        return accuracy_score(y, self.transform(x))


class NumpyEncoder(json.JSONEncoder):
    """
    Special json encoder for numpy types.
    From: https://stackoverflow.com/questions/57269741/typeerror-object-of-type-ndarray-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def cross_validate_local(model, data_x, data_fx, cv, problem_type):
    """Do the same as cross_validate from sklearn, but without needing to pickle the models.

    Args:
        model (_type_): _description_
        data_x (_type_): _description_
        data_fx (_type_): _description_
        cv (_type_): _description_
    """
    scores = []
    fit_times = []

    for train_index, test_index in cv.split(data_x, data_fx):
        X_train, X_test = data_x[train_index], data_x[test_index]
        y_train, y_test = data_fx[train_index], data_fx[test_index]

        if scipy.sparse.issparse(X_train):
            X_train = X_train.todense()
        if scipy.sparse.issparse(X_test):
            X_test = X_test.todense()

        _local_pipeline = model[0]()
        _local_pipeline.fit(X_train, y_train)
        X_train_transformed = _local_pipeline.transform(X_train)
        X_test_transformed = _local_pipeline.transform(X_test)
        model_k = model[1](X_train_transformed, y_train)

        t0 = time()
        model_k.fit(X_train_transformed, y_train)
        t_fit = time()-t0
        y_test_predicted = model_k.predict(X_test_transformed)
        
        if problem_type=='classification':
            score_k = accuracy_score(y_test_predicted, y_test)
        else:
            score_k = np.exp(-np.sqrt(mean_squared_error(y_test, y_test_predicted)) / np.std(y_test))
        
        scores.append(score_k)
        fit_times.append(t_fit)

    return {
        'test_score': scores,
        'fit_time': fit_times
    }


def evaluate_model(model, task, info, random_state=1):
    dataset = datasets.get_dataset(task.dataset_id)
    data_x, data_fx, _, _ = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )
    if len(data_fx.shape) == 1:
        data_fx = data_fx.reshape(-1, 1)
    
    n_max_points = 5000
    _rng = np.random.default_rng(1)
    idx = _rng.choice(data_x.shape[0], size=(min(data_x.shape[0], n_max_points),), replace=False)
    data_x = data_x[idx, :]
    data_fx = data_fx[idx, :]
    
    # Test its performance
    n_splits = 10
    if info['problem_type'] == 'classification':
        grouping = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        grouping = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cross_val_results = cross_validate_local(model, data_x, data_fx, cv=grouping, problem_type=info['problem_type'])

    return {
        'n_layers': info['n_layers'],
        'n_functions': info['n_functions'],
        'cv_mean': np.mean(cross_val_results['test_score']),
        'cv_std': np.std(cross_val_results['test_score']),
        'cv_all': cross_val_results['test_score'],
        'cv_fit_time': cross_val_results['fit_time'],
        'problem_name': task.task_id,
        'problem_type': info['problem_type'],
        'problem_size': data_x.shape[0],
        'problem_features': data_x.shape[1],
    }


def _pipeline_with_preprocessing(task):
    dataset = task.get_dataset()
    features = dataset.features
    categorical_feature_indices = []
    numeric_feature_indices = []
    feature_index = 0
    for i in range(len(features)):
        if features[i].name == task.target_name:
            continue
        if features[i].name == dataset.row_id_attribute:
            continue
        if dataset.ignore_attribute is not None and features[i].name in dataset.ignore_attribute:
            continue
        if features[i].data_type == "nominal":
            categorical_feature_indices.append(feature_index)
        else:
            numeric_feature_indices.append(feature_index)
        feature_index += 1

    pipe = Pipeline(
        steps=[
            (
                "Preprocessing",
                compose.ColumnTransformer(
                    [
                        (
                            "continuous",
                            impute.SimpleImputer(strategy="median"),
                            numeric_feature_indices,
                        ),
                        (
                            "categorical",
                                preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"),
                            categorical_feature_indices,
                        )
                    ]
                )
            ),
            (
                "Scaling",
                compose.ColumnTransformer(
                    [
                        (
                            "continuous-scaler",
                            RobustScaler(),
                            np.arange(0, len(numeric_feature_indices)),
                        ),
                    ],
                    remainder='passthrough'
                )
            )
        ]
    )
    return pipe

def create_sampled_network(k_layers, is_classification, layer_width = 500):
    def _fcn(layer_width, random_seed):
        return Dense(
                 activation='tanh',
                 layer_width=layer_width,
                 random_seed=random_seed,
                 is_classifier=is_classification,
                 parameter_sampler='tanh' )
    def _linear():
        return Linear(is_classifier=is_classification, regularization_scale=1e-5)

    steps = \
    [(f'fcn-{k}', _fcn(layer_width=layer_width, random_seed=k*42)) for k in range(k_layers)] + \
    [('linear', _linear())]

    return Pipeline(steps=steps, verbose=False)

def create_model(task, problem_type='classification', k_layers = 2, n_basis = 150, model_type='sampling', **tf_fit_kwargs):
    def create_model_inner(X_data, y_data):
        if model_type == 'sampling':
            new_model = create_sampled_network(k_layers=k_layers, layer_width=n_basis, is_classification=problem_type=='classification')
        elif model_type == 'tensorflow':
            tf_model = create_sampled_network(k_layers=k_layers, layer_width=n_basis, is_classification=problem_type=='classification')
            tf_model.fit(X_data, y_data)

            callback_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

            first_layer = tf_model.steps[0][1]
            new_model = TensorflowTransformer(
                tensorflow_model=swim_to_keras_model(
                    tf_model,
                    input_shape=(first_layer.weights.shape[0]),
                    set_weights=False,
                    optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                    loss='mse'
                ),
                encoding_map=first_layer.prepare_y,
                decoding_map=first_layer.prepare_y_inverse,
                callbacks=[callback_early_stopping],
                **tf_fit_kwargs
            )
        else:
            raise ValueError(f'Model type {model_type} not defined.')
        return new_model
    return (lambda: _pipeline_with_preprocessing(task=task), create_model_inner), \
            {
                'n_layers': k_layers,
                'n_functions': n_basis,
                'problem_type': problem_type
            }


def write_result(test_result):
    datestr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder = os.path.join(os.getcwd(), 'experiments', 'openml', 'hyperparameter_study', f"{test_result['model_type']}_{test_result['test_id']}")
    os.makedirs(folder, exist_ok=True)
    filepath_result = os.path.join(folder, f"task-{test_result['problem_name']}_{datestr}.json")
    with open(filepath_result, 'w+', encoding='utf8') as file:
        json.dump(test_result, file, cls=NumpyEncoder)


def augment_result_with_info(_result, _info):
    for key in _info.keys():
        _result[key] = _info[key]
    return _result


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    experiment_info_tensorflow = {
        'description': 'Adam',
        'tf_epochs': 100,
        'tf_batch_size': 64,
        'tf_learning_rate': 1e-3,
        'tf_optimizer': 'Adam',
        'model_type': 'tensorflow',
        'n_basis': 500,
        'max_layers': 5
    }
    
    experiment_info_sampling = {
        'description': 'Sampling',
        'activation': 'tf.nn.tanh', # note that this does NOT change the actual activation function above!
        'model_type': 'sampling',
        'n_basis': 500,
        'max_layers': 5
    }
    
    # switch this if you want to run another type of experiment
    experiment_info = experiment_info_sampling

    # ignore some tasks since something is wrong with predict_proba class assignment and weird indexing.
    suite_setups = [
        {
            'suite_id': 99,
            'problem_type': 'classification',
            'ignore_tasks': [] # [45, 3902, 3903, 3904, 3917, 3918, 14954, 146800, 146819, 146822, 146824] # these work now
        }
    ]

    experiment_info['test_id'] = np.random.random_integers(low=100, high=999999, size=(1, ))[0]

    for suite_setup in suite_setups:
        suite = study.get_suite(suite_setup['suite_id'])

        print(suite)
        print('TASKS:', suite.tasks)
        for _k_layers in range(1, experiment_info.get('max_layers', 5)+1, 1):
            for task_id in suite.tasks: # [:5]: # only run five tasks as a test
                if task_id in suite_setup['ignore_tasks']:
                    continue
                try:
                    print(f"Running task {task_id}...")
                    _task = tasks.get_task(task_id)
                    model, info = create_model(
                        _task,
                        k_layers=_k_layers,
                        problem_type=suite_setup['problem_type'],
                        n_basis=experiment_info.get('n_basis', 500),
                        model_type=experiment_info.get('model_type', 'sampling'),
                        epochs=experiment_info.get('tf_epochs', 10),
                        batch_size=experiment_info.get('tf_batch_size', 64)
                    )
                    result = evaluate_model(model, _task, info)
                    result = augment_result_with_info(result, experiment_info)

                    result['suite_id'] = suite_setup['suite_id']
                    result['ignore_tasks'] = suite_setup['ignore_tasks']

                    write_result(result)
                    print(result)
                except FileNotFoundError as e:
                    print(f"File not found for task id {task_id}, {e}.")
