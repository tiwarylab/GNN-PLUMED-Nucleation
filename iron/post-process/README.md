# Files for post-processing.

To run, example command `plumed driver --plumed plumed.dat --mf_xtc md.xtc`.

Output COLVAR file has 5 components:

`#! FIELDS time model.node-0 model.node-1 model.node-2 model.node-3 model.node-4`

model.node-0     $\rightarrow$        other \
model.node-1     $\rightarrow$        FCC \
model.node-2     $\rightarrow$        HCP \
model.node-3     $\rightarrow$        BCC \
model.node-4     $\rightarrow$        ICO 
