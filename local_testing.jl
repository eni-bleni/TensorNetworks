using MPS
using TEBD
using TensorOperations
using Plots
plotlyjs()
println("\n---------------------------------------")

# This file is not tracked by git. Use this to test

#### xxxxxxxxxx

################################################################################
##                                  DMRG
################################################################################

##================================================ Ising model:
latticeSize = 10
J = 1.0
h = 0
g = 0
maxBondDim = 40
prec = 1e-8

hamiltonian = MPS.IsingMPO(latticeSize,J,h,g)
# hamiltonian = MPS.HeisenbergMPO(latticeSize,1,1,1,1)
mps = MPS.randomMPS(latticeSize,2,maxBondDim)

MPS.makeCanonical(mps)

@time ground,E = MPS.DMRG(mps,hamiltonian,prec)

println("E/N = ", E/(latticeSize-1))

@time exc,E = MPS.DMRG(mps,hamiltonian,prec,ground)

println("\nOverlap: ",MPS.MPSoverlap(exc,ground))


# ##================================================ Heisenberg model:
# latticeSize = 21
# Jx = 1.0
# Jy = 1.0
# Jz = 1.0
# h = 0.0
# maxBondDim = 40
# prec = 1e-5
#
# hamiltonian = MPS.HeisenbergMPO(latticeSize,Jx,Jy,Jz,h)
#
# mps = MPS.randomMPS(latticeSize,2,maxBondDim)
#
# mps = MPS.canonicalMPS(mps,1)
#
# @time ground,E = MPS.DMRG2(mps,hamiltonian,prec)
#
# println("E/N = ", E/(latticeSize-1))
# println()



################################################################################
##                                  TEBD
################################################################################
# println("\n...performing TEBD...")
#
# J = 1.0
# h = 0.0
# g = 0.0
#
# N = 10
# dt = 0.01
# D = 20
# d = 2
# steps = 1e3     # number of time evolution steps of size dt
#
# imag_time = 0   # choose between imaginary(1) or real(0) time evolution
#
#
# ## method 2: (as in iTEBD)
# ham = TEBD.TwoSiteIsingHamiltonian(J,h,g)
# if imag_time == 1
#     W = expm(-dt*ham)
# else
#     W = expm(-im*dt*ham)
# end
# W = reshape(W, d,d,d,d)
#
#
# mps = MPS.randomMPS(N,d,D)
# mps = MPS.canonicalMPS(mps,1)
# # mps = ground
# # mps = convert(SharedArray, mps)
#
# println( "E = ", MPS.mpoExpectation(mps, MPS.IsingMPO(N,J,h,g)) )
# println("norm: ", MPS.MPSnorm(mps))
#
# tic()
# for counter = 1:steps
#     for i = 1:2:N-1 # odd sites
#         mps[i], mps[i+1] = TEBD.block_decimation(W, mps[i], mps[i+1], D)
#     end
#
#     for i = 2:2:N-1 # even sites
#         mps[i], mps[i+1] = TEBD.block_decimation(W, mps[i], mps[i+1], D)
#     end
#     if imag_time == 1
#         mps[1] = mps[1]/sqrt(MPS.MPSnorm(mps))
#     end
# end
# toc()
#
# println( "E = ", MPS.mpoExpectation(mps, MPS.IsingMPO(N,J,h,g)) )
# println( "E/N = ", MPS.mpoExpectation(mps, MPS.IsingMPO(N,J,h,g))/(N-1) )
# println("norm: ", MPS.MPSnorm(mps))
