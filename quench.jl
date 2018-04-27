using MPS
using TEBD
#using TensorOperations
using PyPlot
println("\n---------------------------------------")

## parameters for the spin chain:
latticeSize = 10
maxBondDim = 40
d = 2
prec = 1e-8

## Ising parameters:
J  = 1.0
h0 = 1.0
g  = 0.0

## Heisenberg parameters:
Jx = 1.0
Jy = 1.0
Jz = 1.0
hx0 = 1.0

## TEBD parameters:
total_time = 10.0 # -im*total_time  for imag time evol
steps = 1000

# define Pauli matrices:
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]


function isingQuench(i,time)
    ZZ = kron(sz, sz)
    ZI = kron(sz, si)
    IZ = kron(si, sz)
    XI = kron(sx, si)
    IX = kron(si, sx)

    h = h0 + exp(-(time-2)^2)

    if i==1
        return J*ZZ + h/2*(XI+2IX) + g/2*(ZI+2IZ)
    elseif i==latticeSize-1
        return J*ZZ + h/2*(2*XI+IX) + g/2*(2*ZI+IZ)
    else
        return J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ)
    end
end

function heisenbergQuench(i,time)
    XX = kron(sx, sx)
    YY = kron(sy, sy)
    ZZ = kron(sz, sz)
    XI = kron(sx, si)
    IX = kron(si, sx)

    hx = hx0 + exp(-(time-2)^2)

    if i==1
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+2*IX)
    elseif i==latticeSize-1
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(2*XI+IX)
    else
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+IX)
    end
end

hamiltonian = MPS.IsingMPO(latticeSize, J, h0, g)
# hamiltonian = MPS.HeisenbergMPO(latticeSize, Jx, Jy, Jz, hx0)

mps = MPS.randomMPS(latticeSize,d,maxBondDim)
MPS.makeCanonical(mps)
ground,Eground = MPS.DMRG(mps,hamiltonian,prec)


# magnetMPO = MPS.OneSiteMPO(latticeSize, Int(round(latticeSize/2)), [0 1; 1 0])
# magnetization = TEBD.time_evolve(mps2, ham, total_time, steps, maxBondDim, magnetMPO)

## PLOTTING
# plot(abs.(expect[:,1]), real.(expect[:,2]), show=true)

energy = TEBD.time_evolve_mpoham(ground,isingQuench,total_time,steps,maxBondDim,hamiltonian)
# energy = TEBD.time_evolve_mpoham(ground,heisenbergQuench,total_time,steps,maxBondDim,hamiltonian)

## PLOTTING
plot(abs.(energy[:,1]), real.(energy[:,2]))
show()
