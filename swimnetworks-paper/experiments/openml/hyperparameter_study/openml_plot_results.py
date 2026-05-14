import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    experiments = [
        ('tensorflow_787751', 'blue'),
        ('sampling_754879', 'orange')
    ]
    n_plots = 4
    fig, ax = plt.subplots(1, n_plots, figsize=(3.5*n_plots, 3))

    count_ticks = np.arange(0, 20, 2)
    experiment_layers_all = []
    experiment_data_all = []
    experiment_time_all = []
    experiment_desc_all = []

    markers = ['.', 'x']
    markertype = markers[0]

    sampling_blue = (0.0, 101.0 / 255.0, 189.0 / 255.0)
    adam_orange = (227.0 / 255.0, 114.0 / 255.0, 34.0 / 255.0)
    
    for (experiment_name, experiment_color) in experiments:
        filedir = os.path.join(os.getcwd(), 'experiments', 'openml', 'hyperparameter_study', experiment_name)

        experiment_data = pd.DataFrame()
        for dir_info in os.walk(filedir):
            dirpath, dirnames, filenames = dir_info
            for file in filenames:
                with open(os.path.join(dirpath, file), 'r+', encoding='utf8') as json_file:
                    json_content = json.load(json_file)
                    json_content['cv_fit_time_mean'] = np.mean(json_content['cv_fit_time'])
                    json_content.pop('cv_all')
                    json_content.pop('cv_fit_time')

                    experiment_data = pd.concat([
                        experiment_data,
                        pd.DataFrame([json_content])
                    ], ignore_index=True)

        exp_layers_matrix = []
        exp_data_matrix = []
        exp_time_matrix = []
        exp_descriptions = []
        exp_problems = []
        for (_, experiment) in experiment_data.sort_values('problem_name').groupby('n_layers'):
            problem_names = experiment['problem_name'].to_numpy()
            idx = np.arange(len(problem_names))
            exp_data_matrix.append(experiment['cv_mean'].to_numpy())
            exp_layers_matrix.append(experiment['n_layers'].to_numpy())
            exp_time_matrix.append(experiment['cv_fit_time_mean'].to_numpy())
            exp_descriptions.append(experiment['description'])
            exp_problems = experiment['problem_name'].to_numpy()
        exp_layers_matrix = np.row_stack(exp_layers_matrix).T
        exp_data_matrix = np.row_stack(exp_data_matrix).T
        exp_time_matrix = np.row_stack(exp_time_matrix).T
        exp_descriptions = np.array(exp_descriptions)

        print(f"Score {np.mean(exp_data_matrix)} at {experiment_name}")

        bins_data = np.linspace(0, 1, 50)
        bins_time = np.linspace(0, 2, 50)

        experiment_layers_all.append(exp_layers_matrix)
        experiment_data_all.append(exp_data_matrix)
        experiment_time_all.append(exp_time_matrix)
        experiment_desc_all.append(exp_descriptions)

        if markertype == markers[0]:
            markertype = markers[1]
        else:
            markertype = markers[0]
    
    print(f'There are {len(np.unique(problem_names))} classification problems in total.')

    #ax[0].legend()
    ax[1].plot([0, 1], [0, 1], ':')
    ax[1].set_xlabel("Accuracy " + experiment_desc_all[0][0][0])
    ax[1].set_ylabel("Accuracy " + experiment_desc_all[1][0][0])

    # also show the problem number at each point
    x0_data = np.max(experiment_data_all[0], axis=-1)
    x1_data = np.max(experiment_data_all[1], axis=-1)
    differences = []
    for i, word in enumerate(exp_problems):
        x, y = x0_data[i], x1_data[i]
        differences.append(x-y)
        ax[1].scatter(x, y)
        # ax[1].annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', size=9)
    
    bins_differences = np.linspace(-.3, .3, len(differences))
    differences = np.array(differences)
    ax[2].hist(differences, density=False, alpha=1.0, bins=bins_differences, color=sampling_blue, label='Count')
    ax[2].plot([np.mean(differences)]*2, [0, np.max(count_ticks)], ':', label='Average')
    ax[2].set_xlabel(f'Accuracy ({experiment_desc_all[0][0][0]} - {experiment_desc_all[1][0][0]})')
    ax[2].set_ylabel('Count')
    ax[2].set_yticks(count_ticks)
    ax[2].legend()

    x0_data = np.max(experiment_time_all[0], axis=-1)
    x1_data = np.max(experiment_time_all[1], axis=-1)
    bins_differences = np.linspace(np.log(1e-2) / np.log(10), np.log(1e2) / np.log(10), len(x0_data))
    ax[0].hist(np.log(np.abs(x0_data)) / np.log(10), color=adam_orange, density=False, bins=bins_differences, label=f"{experiment_desc_all[0][0][0]}")
    ax[0].hist(np.log(np.abs(x1_data)) / np.log(10), color=sampling_blue, density=False, bins=bins_differences, label=f"{experiment_desc_all[1][0][0]}")
    
    ax[0].plot([np.mean(np.log(np.abs(x0_data)) / np.log(10))]*2, [0, np.max(count_ticks)], ':', color='blue')
    ax[0].plot([np.mean(np.log(np.abs(x1_data)) / np.log(10))]*2, [0, np.max(count_ticks)], ':', color='orange')
    ax[0].set_yticks(count_ticks)

    print(f'Adam is {np.mean(np.abs(x0_data)) / np.mean(np.abs(x1_data)):.1f} times slower on average.')
    
    ax[0].set_xlabel('Fit time (log 10)')
    ax[0].set_ylabel('Count')
    ax[0].legend(loc='upper right')

    # also show the problem number at each point
    if n_plots == 4:
        layer_differences = []
        x0_data = [experiment_layers_all[0][idx][n_layer] for (idx,n_layer) in enumerate(np.argmax(experiment_data_all[0], axis=-1))]
        x1_data = [experiment_layers_all[1][idx][n_layer] for (idx,n_layer) in enumerate(np.argmax(experiment_data_all[1], axis=-1))]
        for i, word in enumerate(exp_problems):
            x, y = x0_data[i], x1_data[i]
            layer_differences.append(x-y)
        bins_differences = np.linspace(-4, 4, len(layer_differences)//2)
        ax[3].hist(layer_differences, density=False, bins=bins_differences, color=sampling_blue)
        ax[3].set_xlabel(f'Layer difference ({experiment_desc_all[0][0][0]} - {experiment_desc_all[1][0][0]})')
        ax[3].set_ylabel('Count')
        ax[3].set_yticks(count_ticks)

    fig.tight_layout()

    figurepath = os.path.join('./experiments', 'openml', 'hyperparameter_study', 'figures')
    os.makedirs(figurepath, exist_ok=True)
    fig.savefig(os.path.join(figurepath, 'openml_sampling_vs_adam.png'))
    fig.savefig(os.path.join(figurepath, 'openml_sampling_vs_adam.pdf'))
