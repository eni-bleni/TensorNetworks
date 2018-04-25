using MPS
using TEBD
using Plots
println("\n---------------------------------------")

latticeSize=10
J=1
h=1.0
g=0.0
maxBondDim = 20
prec=1e-8

total_time = 10.0 # -im*total_time  for imag time evol
steps = 1000

# define Pauli matrices
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]


################################################################################
##                                  DMRG
################################################################################

hamiltonian = MPS.IsingMPO(latticeSize,J,h,g)
# hamiltonian = MPS.HeisenbergMPO(latticeSize,1,1,1,1)

mps = MPS.randomMPS(latticeSize,2,maxBondDim)
# mps2 = MPS.canonicalMPS(mps,1)
MPS.makeCanonical(mps)
println("overlap: ",MPS.MPSoverlap(mps,mps2))
# println(MPS.check_LRcanonical(mps[1],1))

@time ground,E = MPS.DMRG(mps,hamiltonian,prec)

println("E/N = ", E/(latticeSize-1))
println("entropy: ",MPS.entropy(ground,3))

# @time exc,E = MPS.DMRG(mps,hamiltonian,prec,ground)
#
# println("\nOverlap: ",MPS.MPSoverlap(exc,ground))

################################################################################
##                                  TEBD
################################################################################

ham = TEBD.TwoSiteIsingHamiltonian(J,h,g)

mps2 = MPS.randomMPS(latticeSize,2,maxBondDim)
mps2 = ground
MPS.makeCanonical(mps2)

println("Norm: ", MPS.MPSnorm(mps2)," Energy: ", MPS.mpoExpectation(mps2,hamiltonian))

magnetization = MPS.OneSiteMPO(latticeSize, Int(round(latticeSize/2)), sx)
expect = TEBD.time_evolve(mps2, ham, total_time, steps, maxBondDim, magnetization)

println("Norm: ", MPS.MPSnorm(mps2)," Energy: ", MPS.mpoExpectation(mps2,hamiltonian))

## PLOTTING
plot(abs.(expect[:,1]), real.(expect[:,2]), show=true)

;
