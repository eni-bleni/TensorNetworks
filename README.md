# TensorNetworks
Tensor network codes in julia by groups at AEI (GQFI and QGUT)

# How to

In this project, tensor network algorithms for matrix product states are implemented.

i) module MPS:
    contains functions for the initialization of random MPS and MPO
    and the DMRG algorithm for finding ground states and excited states

   functions:



# ToDo

1. Set up the function that computes correlation functions.

(O_i,j_i) -> MPO with O_i's at positions j_i's and id elsewhere. -> compute expectation value of this MPO with existing function.

2. Set up the Quench of the ground state.  
2.1. Compute time dependence of expectation values after/while the quench.

3. Compute thermal state e^(-beta H)

4. Compute reduced density matrices of pure and thermal state.

5. Try to compute as highly excited states as possible.

6. Check subsystem ETH for excited states.

7. Generalize to more Hamiltonians.
