# Files for post-processing.

To run, example command `plumed driver --plumed plumed.dat --mf_xtc md.xtc --mc mcfile`.

Output COLVAR file has 7 components:

`#! FIELDS time model.node-0 model.node-1 model.node-2 model.node-3 model.node-4 model.node-5 model.node-6`

model.node-0     $\rightarrow$        $\ell$-gly \
model.node-1     $\rightarrow$        $\alpha$-gly \
model.node-2     $\rightarrow$        $\beta$-gly \
model.node-3     $\rightarrow$        $\gamma$-gly \
model.node-4     $\rightarrow$        $\delta$-gly \
model.node-5     $\rightarrow$        $\epsilon$-gly \
model.node-6     $\rightarrow$        $\zeta$-gly 
