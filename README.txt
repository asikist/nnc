# General Notes

The present code repository contains code based on the work of the paper titled:
Neural Ordinary Differential Equation Control of Dynamics on Graphs
by authors: 
Thomas Asikis∗
Lucas Böttcher†
Nino Antulov-Fantulin∗
∗ETH Zürich, Switzerland
{asikist, anino}@ethz.com
†Computational Medicine, UCLA, Los Angeles, USA
lucasb@ucla.edu

# Structure
- The ```nnc``` folder contains the main module that has been used to develop thiswork and can be found in:
https://github.com/asikist/nnc
This repo will be actively updated.

- The ```example folder``` contains an illustrative Jupyter notebook with a small scale example of NODEC control.
Please use to get started.

- The ```nodec_experiments``` folder contains all the experimental work and the subfodler contains ```nodec_experiments/data``` contains all experimental parameters and data. The data folder will be provided from a data repository as is larger than 10 GB.
Nevertheless, by running the scripts in the various subfolders of ```nodec_experiments``` can regenrate all data, but it may include some stochasticity and produce slightly varying results in comparison to the paper. Each subfolder contains seperate readme files with more instructions. Jupyter Notebooks also contain a lot of information and commetns on the scripts.
The majority of the code is commented with reStructured docstrings and inline comments.


# Other Notes
Please consider using a GPU.
Some of the scripts take several hours to run even on CUDA enabled machines.
Some unit testing has been done to check some code operations of interest, but the total coverage will increase in the future.

*Any bugs can be reported in the github page as it will updated with the current version.