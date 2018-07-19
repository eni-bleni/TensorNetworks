using MPS
using TEBD
using TensorOperations
using PyPlot
println("\n---------------------------------------")

## parameters for the spin chain:
latticeSize = 10
maxBondDim = 10
d = 2
prec = 1e-8

## Ising parameters:
J0 = 1.0
h0 = 1.0
g0 = 0.0

## Heisenberg parameters:
Jx0 = 1.0
Jy0 = 1.0
Jz0 = 1.0
hx0 = 1.0

## TEBD parameters:
total_time = -im*1.5    # -im*total_time  for imag time evol
steps = 1000
entropy_cut = 4         # subsytem size for entanglement entopy; set to 0 to disregard

# define Pauli matrices:
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]



function thermalIsing(i, time, params) # like isingQuench() function but w/o time evolved params
    J, h, g = params
    ZZ = kron(sz, sz)
    ZI = kron(sz, si)
    IZ = kron(si, sz)
    XI = kron(sx, si)
    IX = kron(si, sx)

    if i==1
        return J*ZZ + h/2*(XI+2IX) + g/2*(ZI+2IZ)
    elseif i==latticeSize-1
        return J*ZZ + h/2*(2*XI+IX) + g/2*(2*ZI+IZ)
    else
        return J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ)
    end
end

function thermalHeisenberg(i,time, params)
    Jx, Jy, Jz, hx = params
    XX = kron(sx, sx)
    YY = kron(sy, sy)
    ZZ = kron(sz, sz)
    XI = kron(sx, si)
    IX = kron(si, sx)

    if i==1
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+2*IX)
    elseif i==latticeSize-1
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(2*XI+IX)
    else
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+IX)
    end
end

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

states,energies = MPS.n_lowest_states(mps, hamiltonian, prec,15)
println("energies: ", energies)


################################################################################
##                                  ETH
################################################################################
println("\n...performing ETH...")

## thermal state MPO:
init_params = (J0, h0, g0)
tr_dist = Array{Any}(length(energies))

tic()
for i = 2:length(energies)
    println("\ni = ", i)
    exc = states[i]
    E1 = energies[i]
    ETH = (true,E1,hamiltonian)
    IDmpo = MPS.IdentityMPO(latticeSize,d)
    E_thermal, betahalf = TEBD.time_evolve_mpoham(IDmpo,thermalIsing,total_time,steps,maxBondDim,1,0,init_params,ETH)
    rho_th = MPS.multiplyMPOs(IDmpo,IDmpo) # = exp[-beta/2 H]*exp[-beta/2 H]'
    tr_dist[i] = MPS.traceMPO(IDmpo,4) -2*MPS.mpoExpectation(exc,rho_th) + 1 # = Tr(rho_th^2) - 2<exc|rho_th|exc> + 1 = Tr([rho_th-|exc><exc|]^2)
    println("E_thermal, beta/2 = ", E_thermal, ", ", betahalf)
    println("E_exc, Tr(dist^2) = ", E1, ", ", real(tr_dist[i]))
end
toc()

## PLOTTING
figure(1)
plot(real(energies[2:end]), real(tr_dist[2:end]))




show()
;
