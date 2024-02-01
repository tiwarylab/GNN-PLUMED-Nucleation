# GNN-PLUMED-Nucleation

Inputs for well-tempered metadynamics simulations biasing along GNN-learned reaction coordinates. Details can be found in [this manuscript](http://arxiv.org/abs/2310.07927). Here we present a simplified procedure for constructing GNN RCs.

- Train and save the GNN model with [Pytorch](https://pytorch.org/). The GNN models used in this work can be found in [this github](https://github.com/mys007/ecc/tree/release). For the two systems, we provided one ready-to-run model as `model.pt` for reproducing the molecular dynamics data.

- Convert the saved Pytorch model into libtorch version so PLUMED can take it as inputs. This is achieved via [```jit.trace```](https://pytorch.org/docs/stable/jit.html#) function.

- Perform [well-tempered metadynamics](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.100.020603) with MD engines patched with PLUMED. PLUMED needs to be compiled with [`PYTORCH_MODEL`](https://mlcolvar.readthedocs.io/en/latest/plumed.html) module. Instructions are provided in [this colab tutorial](https://colab.research.google.com/drive/1dG0ohT75R-UZAFMf_cbYPNQwBaOsVaAA) and in [PLUMED page](https://www.plumed.org/doc-v2.9/user-doc/html/_p_y_t_o_r_c_h__m_o_d_e_l.html).
