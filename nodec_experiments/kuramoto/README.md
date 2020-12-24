# Instructions

1. Please make sure that the required data folder is available at the paths used by the script.
You may generate the required data by running the python script
```nodec_experiments/kuramoto/gen_parameters.py```.


2. The plots use the training results.
Please also make sure that a training proceedure has produced results in the corresponding paths used in plot and table scripts.
Running ```nodec_experiments/kuramoto/train.ipynb``` with default paths is expected to generate at the requiered location for the plots and table scripts in each folder.

3. Multi sample evaluation plots also require:
`nodec_experiments/kuramoto/multi_sample/evaluate_initial_states.ipynb`

As neural network intialization is stochastic, please make sure that appropriate seeds are used or expect some variance to paper results.

