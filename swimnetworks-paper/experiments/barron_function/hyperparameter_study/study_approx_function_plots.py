import numpy as np
import matplotlib.pyplot as plt

import os
import pandas as pd

if __name__ == '__main__':
    outputfile_label = 'BarronE'
    outputfile_method = 'sampling'
    filepath = './experiments/barron_function/hyperparameter_study/'
    filenames = [
        #os.path.join(filepath, 'experiment_BarronE_sampling_with_tanh_10000pts_5247321.csv'), # sampling, with tanh
        os.path.join(filepath, 'experiment_BarronE_randomfeatures_10000pts_3047705.csv'), # randomfeatures, with sin
        os.path.join(filepath, 'experiment_BarronE_sampling_with_sin_10000pts_1561250.csv'), # sampling, with sin
    ]
    experiment_df = pd.concat([
        pd.read_csv(filename)
        for filename in filenames
    ])

    n_layers_total = 1 #3
    n_dims = len(experiment_df.groupby('n_dim'))
    k_dim = 0

    fig_l2, ax_l2 = plt.subplots(n_dims, n_layers_total, figsize=(6, n_dims*3), sharey=False)
    fig_time, ax_time = plt.subplots(n_dims, n_layers_total, figsize=(6, n_dims*3), sharey=False)
    
    if np.array(ax_l2).ndim < 2:
        ax_l2 = np.array(ax_l2).reshape((-1, 1))
        ax_time = np.array(ax_time).reshape((-1, 1))

    color_dict = {
        'adam': (227.0 / 255.0, 114.0 / 255.0, 34.0 / 255.0),
        'randomfeatures': (227.0 / 255.0, 114.0 / 255.0, 34.0 / 255.0), # (162.0 / 255., 173./255., 0.0),
        'sampling': (0.0, 101.0 / 255.0, 189.0 / 255.0),
    }

    layer_marker = {
        1: '-.',
        2: '-x',
        3: ':o',
    }

    for (n_dim, ndim_df) in experiment_df.groupby('n_dim'):
        legends = []
        n_layer_plots = 0
        for (n_layers, df) in ndim_df.groupby('n_layers'):
            k2 = 0
            for (model_type, df_layer) in df.groupby('model_type'):
                df_layer.groupby('n_width').mean(numeric_only=True).reset_index().plot(
                    x='n_width',
                    y='l2_relative',
                    logy=True,
                    logx=True,
                    ax=ax_l2[k_dim, k2],
                    color=color_dict[model_type],
                    style=layer_marker[n_layers]
                )
                legends.append(f"{model_type}, L={n_layers}")
                ax_l2[k_dim, k2].set_xlabel(f'Network width')
                if n_layers==1:
                    ax_l2[k_dim, k2].set_ylabel(fr'Rel. $~L^2$ error, dim={n_dim}')

            if n_layer_plots == 2:
                legends.append(r'reference $m^{-1/2}$')
                legends.append(r'reference $m^{-1}$')
                convergence_df = df[df['model_type'] == 'sampling']
                reference_convergence1 = np.power(convergence_df['n_width'], -1/2.0)
                reference_convergence1 /= np.max(reference_convergence1)
                reference_convergence1 *= np.max(convergence_df['l2_relative'])
                reference_convergence2 = np.power(convergence_df['n_width'], -1.0)
                reference_convergence2 /= np.max(reference_convergence2)
                reference_convergence2 *= np.max(convergence_df['l2_relative'])
                ax_l2[k_dim, k2].plot(convergence_df['n_width'], reference_convergence1, ':', color='black')
                ax_l2[k_dim, k2].plot(convergence_df['n_width'], reference_convergence2, '--', color='black')
                
                #ax_l2[k_dim, k2].set_ylim([1e-5, 1e0])
            n_layer_plots += 1
        ax_l2[k_dim, 0].legend(legends, bbox_to_anchor=(1,1), loc="upper left")

        legends = []
        for (n_layers, df) in ndim_df.groupby('n_layers'):
            for (model_type, df_layer) in df.groupby('model_type'):
                df_layer.groupby('n_width').mean(numeric_only=True).reset_index().plot(
                    x='n_width',
                    y='time_fit',
                    logy=True,
                    logx=True,
                    ax=ax_time[k_dim, k2],
                    color=color_dict[model_type],
                    style=layer_marker[n_layers]
                )
                legends.append(f"{model_type}, L={n_layers}")
                ax_time[k_dim, k2].set_xlabel('Network width')
                if n_layers==1:
                    ax_time[k_dim, k2].set_ylabel(fr'Fit time [s], dim={n_dim}')
        ax_time[k_dim, k2].legend(legends, bbox_to_anchor=(1,1), loc="upper left")
        k_dim += 1
    fig_l2.tight_layout()
    fig_time.tight_layout()
    os.makedirs(os.path.join(filepath, 'figures'), exist_ok=True)
    fig_l2.savefig(os.path.join(filepath, 'figures', f'{outputfile_label}_ndim_{n_dims}_l2_by_modeltype_onlysin.png'))
    fig_time.savefig(os.path.join(filepath, 'figures', f'{outputfile_label}_ndim_{n_dims}_time_by_modeltype_onlysin.png'))
    fig_l2.savefig(os.path.join(filepath, 'figures', f'{outputfile_label}_ndim_{n_dims}_l2_by_modeltype_onlysin.pdf'))
    fig_time.savefig(os.path.join(filepath, 'figures', f'{outputfile_label}_ndim_{n_dims}_time_by_modeltype_onlysin.pdf'))
