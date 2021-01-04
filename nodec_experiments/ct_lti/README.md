# Instructions

## Script sequence

Please make sure that the required data folder is available at the paths used by the script.
You may generate the required data by running the python script:
`nodec_experiments/ct_lti/gen_parameters.py`.

Please also make sure that a training proceedure has produced results in the corresponding paths used in plot and table scripts.
Running for training the following scripts for training and evaluation:
- single sample training and evaluation: `nodec_experiments/ct_lti/single_sample/train.ipynb`
- evaluation of model parameters per 5 epochs, on data stored during training:  `nodec_experiments/ct_lti/single_sample/figure_4_evaluate.ipynb`
- multi-sample training and evaluation which takes 15 hours per graph:  `nodec_experiments/ct_lti/single_sample/train.ipynb`
 
Please copy to the default paths under `/data/results/` the required data for the plots and table scripts in each folder.

As neural network intialization is stochastic, please make sure that appropriate seeds are used or expect some variance to paper results.