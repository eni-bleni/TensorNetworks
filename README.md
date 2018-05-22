# TensorNetworks
Tensor network codes in julia by groups at AEI (GQFI and QGUT)

# How to

In this project, tensor network algorithms for matrix product states are implemented.

i) module MPS:
    contains functions for the initialization of random MPS and MPO
    and the DMRG algorithm for finding ground states and excited states

   functions:

function LRcanonical(M,dir)
  -> returns the left or right canonical form of a single MPS tensor
  input:
  M: tensor of size (D1,d,D2)
  dir: direction, -1 is leftcanonical, 1 is rightcanonical
  output:
  A,R,DB: matrices A and R from qr decomposition and intermediate bond dimension DB

function OneSiteMPO(L, j, op)
  -> returns a MPO of length L with identities at each site and operator 'op' at site j
     e.g. for magnetization: op = sx (Pauli matrix)

function IsingMPO(L, J, h, g)
-> returns the Hamiltonian for the Ising model in transverse field as an MPO
input:   
L: lenght of mpo = number of sites/tensors
J,h,g: Ising Hamiltonian params
output:
mpo: constructs Hamiltonian sites of size (a,i,j,b) -> a,b: bond dims, i,j: phys dims
first site: (1,i,j,b); last site: (a,i,j,1)



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

8. Improve efficiency of DMRG: The use of Heff as a (large) matrix in eigs can (and should) be improved by defining a linear map directly in terms of HL*MPO*HR. The packages https://github.com/Jutho/LinearMaps.jl or https://github.com/JuliaSmoothOptimizers/LinearOperators.jl can be used to construct a function that can be used in eigs.
