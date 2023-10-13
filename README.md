# GNN-PLUMED-Nucleation

Inputs for well-tempered metadynamics simulations biasing along GNN-learned reaction coordinates. Details can be found in [this manuscript](http://arxiv.org/abs/2310.07927). Here we present a simplified procedure for constructing GNN RCs.

- Train and save your GNN model using [pytorch](https://pytorch.org/). 

- Convert the saved pytorch model into libtorch version so PLUMED can take it as inputs. This is achieved via [```jit.trace```](https://pytorch.org/docs/stable/jit.html#) function.

- Perform well-tempered metadynamics with MD engines patched with PLUMED. PLUMED needs to be compiled with Pytorch module. Instructions are provided in [this colab tutorial](https://colab.research.google.com/drive/1dG0ohT75R-UZAFMf_cbYPNQwBaOsVaAA).
