using MPS
using TEBD
using Plots
println("\n---------------------------------------")

latticeSize=10
J=1
h=1.0
g=0.0
maxBondDim = 200
prec=1e-8

total_time = 10.0
steps = 1000

################################################################################
##                                  DMRG
################################################################################

hamiltonian = MPS.IsingMPO(latticeSize,J,h,g)
# hamiltonian = MPS.HeisenbergMPO(latticeSize,1,1,1,1)

mps = MPS.randomMPS(latticeSize,2,maxBondDim)
# mps2 = MPS.canonicalMPS(mps,1)
MPS.makeCanonical(mps)
# println("overlap: ",MPS.MPSoverlap(mps,mps2))
# println(MPS.check_LRcanonical(mps[1],1))

@time ground,E = MPS.DMRG(mps,hamiltonian,prec)

println("entropy: ",MPS.entropy(ground,3))

@time exc,E = MPS.DMRG(mps,hamiltonian,prec,ground)

println("Overlap: ",MPS.MPSoverlap(exc,ground))


################################################################################
##                                  TEBD
################################################################################

ham = TEBD.TwoSiteIsingHamiltonian(J,h,g)
mps2 = MPS.randomMPS(latticeSize,2,maxBondDim)
# MPS.makeCanonical(mps2)
println("Norm: ", MPS.MPSnorm(mps2)," Energy: ", MPS.mpoExpectation(mps2,hamiltonian))
expect = TEBD.time_evolve(mps2, ham, -im*total_time, steps, maxBondDim, hamiltonian)
println("Norm: ", MPS.MPSnorm(mps2)," Energy: ", MPS.mpoExpectation(mps2,hamiltonian))

## PLOTTING
plot(abs.(expect[:,1]), real.(expect[:,2]), show=true)

;
