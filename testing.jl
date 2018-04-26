using MPS
using TEBD
using Plots
println("\n---local------------------------------------")

# This file is not tracked by git. Use this to test
#+++++++++++++++++++++++++++++++++++++++++++++++++++


## parameters for the spin chain:
latticeSize = 100
maxBondDim = 20
d = 2
prec = 1e-8

## Ising parameters:
J = 1
h = 1.0
g = 0.0

## Heisenberg parameters:
Jx = 1.0
Jy = 1.0
Jz = 1.0
hx = 1.0

## TEBD parameters:
total_time = -im*10.0 # -im*total_time  for imag time evol
steps = 1000

# define Pauli matrices:
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]



################################################################################
##                                  DMRG
################################################################################
println("...performing ground state DMRG...")

# hamiltonian = MPS.IsingMPO(latticeSize, J, h, g)
hamiltonian = MPS.HeisenbergMPO(latticeSize, Jx, Jy, Jz, hx)

mps = MPS.randomMPS(latticeSize,d,maxBondDim)
MPS.makeCanonical(mps)

@time ground,E = MPS.DMRG(mps,hamiltonian,prec)
println("E/N = ", E/(latticeSize-1))
# println("entropy: ",MPS.entropy(ground,3))

println("\n...performing excited state DMRG...")
@time exc,E = MPS.DMRG(mps,hamiltonian,prec,ground)
println("Overlap: ",MPS.MPSoverlap(exc,ground))



################################################################################
##                                  TEBD
################################################################################
# println("\n...performing TEBD...")
#
# # ham = TEBD.TwoSiteIsingHamiltonian(J,h,g)
# ham = TEBD.TwoSiteHeisenbergHamiltonian(Jx,Jy,Jz,hx)
#
# mps2 = MPS.randomMPS(latticeSize,d,maxBondDim)
# # mps2 = ground
# MPS.makeCanonical(mps2)
#
# println("Norm: ", MPS.MPSnorm(mps2)," Energy: ", MPS.mpoExpectation(mps2,hamiltonian))
#
# magnetization = MPS.OneSiteMPO(latticeSize, Int(round(latticeSize/2)), sx)
# expect = TEBD.time_evolve(mps2, ham, total_time, steps, maxBondDim, hamiltonian)
#
# println("Norm: ", MPS.MPSnorm(mps2)," Energy: ", MPS.mpoExpectation(mps2,hamiltonian))
# println( "E/N = ", MPS.mpoExpectation(mps2,hamiltonian)/(latticeSize-1) )
#
#
# ## PLOTTING
# plot(abs.(expect[:,1]), real.(expect[:,2]), show=true)

;
