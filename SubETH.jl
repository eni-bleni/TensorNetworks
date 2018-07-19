using MPS
using TEBD
using TensorOperations
using PyPlot
println("\n---------------------------------------")

## parameters for the spin chain:
latticeSize = 30
maxBondDim = 20
d = 2
prec = 1e-8


## Ising parameters:
J0 = 1.0
h0 = 1.0
g0 = 0.0

## Heisenberg parameters:
#Jx0 = 1.0
#Jy0 = 1.0
#Jz0 = 1.0
#hx0 = 1.0

thermhamblocks(time) = TEBD.isingHamBlocks(latticeSize,J0,h0,g0)

## TEBD parameters:
total_time = -im*3    # -im*total_time  for imag time evol
steps = 2000
entropy_cut = 0         # subsytem size for entanglement entopy; set to 0 to disregard



################################################################################
##                                  DMRG
################################################################################
println("...performing ground state DMRG...")

hamiltonian = MPS.IsingMPO(latticeSize, J0, h0, g0)
# hamiltonian = MPS.HeisenbergMPO(latticeSize, Jx0, Jy0, Jz0, hx0)

mps = MPS.randomMPS(latticeSize,d,maxBondDim)
MPS.makeCanonical(mps)

# ground,E0 = MPS.DMRG(mps,hamiltonian,prec)
#
#
# println("\n...performing excited state DMRG...")
# exc,E1 = ground,E0
# while E1 ≈ E0 # find true excited state, not degenerate ground state
#     exc,E1 = MPS.DMRG(mps,hamiltonian,prec,ground)
# end

states,energies = MPS.n_lowest_states(mps, hamiltonian, prec,10)
println("energies: ", energies)


################################################################################
##                                  Subsystem ETH
################################################################################
println("\n...performing ETH...")

## thermal state MPO:
init_params = (J0, h0, g0)
sub_tr_dist = Array{Complex64}(length(energies))
subSize = 30
tic()
for i = 2:length(energies)
    println("\ni = ", i)
    exc = states[i]
    E1 = energies[i]
    Rho = MPS.IdentityMPO(latticeSize,d)
    Ethermal, betahalf = TEBD.tebd_simplified(Rho,thermhamblocks,total_time,steps,maxBondDim,[],0,(E1,hamiltonian)) # Rho = exp(-beta/2 H) after this with E_thermal ~ E_exc
    sub_tr_dist[i] = MPS.SubTraceDistance(Rho,exc,subSize,2)
    println("E_thermal, E_exc, beta = ", real(Ethermal), ", ", E1, ",", 2*betahalf)
    println("SubTrDist = ", real(sub_tr_dist[i]))
end
toc()

## PLOTTING
figure(1)
plot(real(energies[2:end]), real(sub_tr_dist[2:end]))




show()
;
