# Instructions

## Script sequence

Please make sure that the required data folder is available at the paths used by the script.
You may generate the required data by running the python script
```nodec_experiments/ct_lti/gen_parameters.py```.

Please also make sure that a training proceedure has produced results in the corresponding paths used in plot and table scripts.
Running ```nodec_experiments/ct_lti/single_sample/train.ipynb``` or ```nodec_experiments/ct_lti/single_sample/train.ipynb``` with default paths is expected to generate at the requiered location for the plots and table scripts in each folder.

As neural network intialization is stochastic, please make sure that appropriate seeds are used or expect some variance to paper results.
