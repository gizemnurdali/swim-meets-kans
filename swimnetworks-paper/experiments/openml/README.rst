This experiment concerns the OpenML tasks in the ``OpenML-CC18 Curated Classification benchmark`` (suite_id 99).
We explicitly set up the 10-fold cross-validation, and run over 1-5 layers with 500 neurons each to find the most suitable architecture.

Run "hyperparameter_study/openml_run_study.py" to create data, and then "hyperparameter_study/openml_plot_results.py" to plot the results.
The data used to compare sampled networks and the same, trained with the Adam optimizer, is already available.
