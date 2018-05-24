using MPS
using TEBD
using TensorOperations
using PyPlot
println("\n---------------------------------------")

## parameters for the spin chain:
latticeSize = 10
maxBondDim = 20
d = 2
prec = 1e-8

## Ising parameters:
J0 = 1.0
h0 = 0.0
g0 = 3.0

## Heisenberg parameters:
Jx0 = 1.0
Jy0 = 1.0
Jz0 = 1.0
hx0 = 1.0

## TEBD parameters:
total_time = -im*10.0    # -im*total_time  for imag time evol
steps = 1000
entropy_cut = 4         # subsytem size for entanglement entopy; set to 0 to disregard

# define Pauli matrices:
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]



function isingQuench(i, time, params)
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

function heisenbergQuench(i,time, params)
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

ground,E0 = MPS.DMRG(mps,hamiltonian,prec)


println("\n...performing excited state DMRG...")
E1 = E0
while E1 ≈ E0 # find true excited state, not degenerate ground state
    exc,E1 = MPS.DMRG(mps,hamiltonian,prec,ground)
end


################################################################################
##                                  ETH
################################################################################

## thermal state MPO:
init_params = (J0, h0, g0)
ETH = (true,E1,hamiltonian)
IDmpo = MPS.IdentityMPO(latticeSize,d)
@time E_thermal, betahalf = TEBD.time_evolve(IDmpo,isingQuench,total_time,steps,maxBondDim,0,init_params,ETH)
# rho = MPS.multiplyMPOs(IDmpo,IDmpo)
# E_thermal = MPS.traceMPO(MPS.multiplyMPOs(hamiltonian,rho))
println("E_thermal, beta/2 = ", E_thermal, ", ", betahalf)

;
