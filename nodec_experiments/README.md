# Instructions

In htios folder you may find the experiments that use the nnc module to perform NODEC control and compare it with baselines on different dynamics and graphs.
Pleae use the accompanied paper for more information.

## Existing Data
Existing data for interactive plots based on the paper results are found under the folder data, which is provided seperately in a different repository (it contains several GB of data).

## Data Generation
If the folder is empty then all the data and experiments should be repoducible by running the scripts in this folder.
PLEASE MAKE SURE that path locations are aligned between scripts and also check whether the proper device is used when interacting with torch. Most python training scrips are cleaner and more explanatory versions of python scripts and have not been tested on CPU machines.

### General Proceedure
For generating data and running the plot scripts the following proceedure is implemented. The README.txt files found in the subfolders of this document parent contain more explicit instructions.
1. Please make sure that the required data folder is available at the paths used by the script.
You may generate the required data by running the python script
```nodec_experiments/{experiment}/gen_parameters.py```.

2. The plots may use the training and evaluation results.
Running ```nodec_experiments/{experiments}/nodec_train.ipynb``` and ```nodec_experiments/{experiments}/nodec_train.ipynb```with default paths is expected to generate at the requiered location for the plots and table scripts in each folder.

3. Sample evaluation is done across alla baseliens before running the plots that also require the following script to run:
`nodec_experiments/{experiments}/eval_baselines.ipynb`.\

4. Kuramoto and NODEC offer evaluations and training proceedure over one or more samples in accordance to the paper.

5. Extra scripts on experiments that did not produce good results may not be provide for the sake of space and brevity, e.g. RL-SAC. Still it may be trivial to adjust the existing scripts and parametrize them.

5. Utility scripts also exist to offer dedicated helper functions or classes per dynamics:
lease make sure that they are available in the python path when running experiments.

Reinforcement Learning requires some significant time to train.
Evaluations and training of NODEC over many samples is also time consuming.

We request that you keep in mind always:

As neural network intialization is stochastic, please make sure that appropriate seeds are used or expect some variance to paper results.

Folders named as 'plot_original` were created to contain  the original paper plots.

These scripts use plotly and orca or kaleido to generate static images. If you have trouble using those commands, please disable them. More on:

https://plotly.com/python/static-image-export/
