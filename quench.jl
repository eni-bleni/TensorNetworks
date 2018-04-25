using MPS
using TEBD
using TensorOperations
using Plots
prec = 1e-8
D = 10
L = 10
function isingQuench(i,time)
    return TEBD.TwoSiteIsingHamiltonian(1,sin(time),0)
end
hamiltonian = MPS.IsingMPO(L,1,0,0)
mps = MPS.randomMPS(L,2,D)
MPS.makeCanonical(mps)
ground,Eground = MPS.DMRG(mps,hamiltonian,prec)


# magnetMPO = MPS.OneSiteMPO(latticeSize, Int(round(latticeSize/2)), [0 1; 1 0])
# magnetization = TEBD.time_evolve(mps2, ham, total_time, steps, maxBondDim, magnetMPO)

## PLOTTING
# plot(abs.(expect[:,1]), real.(expect[:,2]), show=true)


energy = TEBD.time_evolve_mpoham(ground,isingQuench,1,1000,D,hamiltonian)
plot(abs.(energy[:,1]), real.(energy[:,2]), show=true)
