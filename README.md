# Neural Network Control (NNC)
A framework for neural network control of dynamical systems over graphs.
The current repository is based on the work presented in the  paper:
- Asikis, T., Böttcher, L., & Antulov-Fantulin, N. (2020). NNC: Neural-Network Control of Dynamical Systems on Graphs. [arXiv](https://arxiv.org/abs/2006.09773) preprint arXiv:2006.09773.
Please cite the paper if this repository is useful for your research.

NNC aims to control dynamical systems that describe the evolution and interactions of networked components.
We refer to these components as state variables on a graph.
A graph consists of nodes that are conncected with edges.
Each node is assigned a state variable value, and this value evolves through time according to some interaction rules that are described in the form of an 
ordinary differential equation.
External control signals may be applied to a subset of nodes to guide the evolution of the system towards a target state.
The main idea is to use neural networks, and more specifically Neural ODEs to learn the aforementioned control signals in a non-supervised manner.

## Examples

For now NNC works only with pytorch, and we offer a very simple yet descriptive example on continuous time time-invariant dynamics, which can be found in:
- `examples/ct_lti_small/small_example.ipynb` or `examples/ct_lti_small/small_example.ipynb`
- [Google Collab](https://colab.research.google.com/github/asikist/nnc/blob/master/examples/ct_lti_small/small_example.ipynb)

The repository will be updated regularly with latest research work and applications.
Our aim is to showcase the capabilities of NNC, especially for high-dimensional non-linear dynamics.

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
