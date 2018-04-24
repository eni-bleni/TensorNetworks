using MPS

latticeSize=20
J=1
h=0.2
g=0.8
maxBondDim = 20
prec=1e-8

hamiltonian = MPS.IsingMPO(latticeSize,J,h,g)

# hamiltonian = MPS.HeisenbergMPO(latticeSize,1,1,1,1)

mps = MPS.randomMPS(latticeSize,2,maxBondDim)

# mps2 = MPS.canonicalMPS(mps,1)
MPS.makeCanonical(mps)
# println("overlap: ",MPS.MPSoverlap(mps,mps2))
# println(MPS.check_LRcanonical(mps[1],1))
MPS.println(mps)
@time ground,E = MPS.DMRG(mps,hamiltonian,prec)
println("entropy: ",MPS.entropy(ground,3))
@time exc,E = MPS.DMRG(mps,hamiltonian,prec,ground)

println("Overlap: ",MPS.MPSoverlap(exc,ground))


ham = MPS.TwoSiteIsingHamiltonian(J,h,g)
mps2 = MPS.randomMPS(latticeSize,2,maxBondDim)
# MPS.makeCanonical(mps2)
println("Norm: ", MPS.MPSnorm(mps2)," Energy: ", MPS.mpoExpectation(mps2,hamiltonian))
MPS.TEBD(mps2,ham,0.1,1000,maxBondDim)
println("Norm: ", MPS.MPSnorm(mps2)," Energy: ", MPS.mpoExpectation(mps2,hamiltonian))
