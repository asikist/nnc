# NNC: Neural Network Control 

The `nnc` module aims to offer neural network controllers in pytorch, such as neural PIDs, Model Predictive control and other baselines.

## NODEC: Neural Ordinary Differential Equation Control

NODEC is a novel method that controls dynamical systems that describe the evolution and interactions of networked components.
We refer to these components as state variables on a graph.
A graph consists of nodes that are conncected with edges.
Each node is assigned a state variable value, and this value evolves through time according to some interaction rules that are described in the form of an 
ordinary differential equation.
External control signals may be applied to a subset of nodes to guide the evolution of the system towards a target state.
The main idea is to use neural networks, and more specifically Neural ODEs to learn the aforementioned control signals in a non-supervised manner.

## Examples

For now NNC and NODEC work only with pytorch, and we offer a very simple yet descriptive example on continuous time time-invariant dynamics, which can be found in:
- `examples/ct_lti_small/small_example.ipynb` or `examples/ct_lti_small/small_example.ipynb`
- [Google Collab](https://colab.research.google.com/github/asikist/nnc/blob/master/examples/ct_lti_small/small_example.ipynb)

The repository will be updated regularly with latest research work and applications.
Our aim is to showcase the capabilities of NODEC, especially for high-dimensional non-linear dynamics.

## Requirements
Requirements to run nnc and its examples.

### Software:

Module dependencies (`nnc/*`):
- Python 3.6+
- torch 1.4+
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
- numpy 1.17.2+
- scipy =1.3.1+

Example dependencies (`examples/*`):
- plotly 4+
- tqdm
- pandas

### Hardware
We tested most our experiments on both CUDA and CPU.
We advise using CUDA with more than 8GB of VRAM for training of NODEC.

## Project structure
In code you may find 4 folders, and their contents are described as:
- `nnc` the main neural network control module with utilities and baselines for neural network control.
- `nodec_experiments` the folder containing the scripts that train and evaluate NODEC vs other baselines.
- `test`: a folder with unit tests on some methods, which in the future will expand to achieve full coverage.
- `examples`: a folder with unit tests, which in the future will expand to achieve full coverage of the project.
- `../data`: a folder that contains the parameter and result data used by the scripts in the `experiments` folder to generate plots, train or evaluate.
- `../results`: a folder that contains the output of the scripts from `nodec_experiments` folder. In case you need to use data from the `results` folder please copy paste the date in the coresponding `data` folder or change the paths in the provided scripts.

## Further Notes

As neural networks rely on stochastic initialization, results may vary compared to those described in our paper if a different weight initialization is used.
Our current seed settings do not replicate exact results on different machines,
but with a few runs it is possible to generate similar results to the paper.
The seed setting will be further examined in the future.

For an overview of NODEC and its application to various dynamical systems, see
our arXiv preprints [arXiv:2006.09773](https://arxiv.org/abs/2006.09773) and [arXiv:2103.06525](https://arxiv.org/abs/2103.06525). 

Data of the numerical experiments can be found at [IEEEDataPort](http://ieee-dataport.org/3452).

These data contain pretrained models and evluation results.

Please cite our work if this repository is useful for your research.

```
@article{asikis2020nnc,
  title={Neural Ordinary Differential Equation Control of Dynamics on Graphs},
  author={Asikis, Thomas and B{\"o}ttcher, Lucas and Antulov-Fantulin, Nino},
  journal={arXiv preprint arXiv:2006.09773},
  year={2020},
  url = {https://arxiv.org/abs/2006.09773}
}
@article{boettcher2021implicit,
  title={Implicit energy regularization of neural ordinary-differential-equation control},
  author={B{\"o}ttcher, Lucas and Antulov-Fantulin, Nino and Asikis, Thomas},
  journal={arXiv preprint arXiv:2103.06525},
  year={2021},
  url={https://arxiv.org/abs/2103.06525}
}
```


