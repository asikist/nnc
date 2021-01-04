# Instructions

1. Please make sure that the required data folder is available at the paths used by the script.
You may generate the required data by running the python script
```nodec_experiments/sirx/gen_parameters.py```.

2. The plots use the training results.
Please also make sure that a training proceedures for both RL and NODEC have produced results in the corresponding paths used in plot and table scripts.
Running ```nodec_experiments/sirx/nodec_train.ipynb``` and ```nodec_experiments/sirx/nodec_train.ipynb```with default paths is expected to generate at the requiered location for the plots and table scripts in each folder.

3. Sample evaluation is done across alla baseliens before running the plots that also require the following script to run:
`nodec_experiments/sirx/eval_baselines.ipynb`

4. Extra scripts on experiments that did not produce good results may not be provide for the sake of space and brevity.

5. The scripts below:
 - ```nodec_experiments/sirx/sirx.py```
 - ```nodec_experiments/sirx/rl_utils.py```
 - ```nodec_experiments/sirx/sirx_utils.py```
contain very important utilities for running training , evaluation and plotting scripts. Please make sure that they are available in the python path when running experiments.

Reinforcement Learning requires some significant time to train.

As neural network intialization is stochastic, please make sure that appropriate seeds are used or expect some variance to paper results.

