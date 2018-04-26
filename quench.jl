using MPS
using TEBD
using TensorOperations
using Plots
prec = 1e-8
D = 100
L = 20
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]
J=1
h0 = 1
function isingQuench(i,time)
    ZZ = kron(sz, sz)
    ZI = kron(sz, si)
    IZ = kron(si, sz)
    XI = kron(sx, si)
    IX = kron(si, sx)
    h = h0 + 5*exp(-10(time-2)^2) #sin(time) *time^2*(pi - time)^2
    g = 0
    if i==1
        return J*ZZ + h/2*(XI+2IX) + g/2*(ZI+2IZ)
    elseif i==L-1
        return J*ZZ + h/2*(2*XI+IX) + g/2*(2*ZI+IZ)
    else
        return J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ)
    end
end
hamiltonian = MPS.IsingMPO(L,J,h0,0)

mps = MPS.randomMPS(L,2,D)
MPS.makeCanonical(mps)
ground,Eground = MPS.DMRG(mps,hamiltonian,prec)


# magnetMPO = MPS.OneSiteMPO(latticeSize, Int(round(latticeSize/2)), [0 1; 1 0])
# magnetization = TEBD.time_evolve(mps2, ham, total_time, steps, maxBondDim, magnetMPO)

## PLOTTING
# plot(abs.(expect[:,1]), real.(expect[:,2]), show=true)

# energy = TEBD.time_evolve_mpoham(ground,isingQuench,-10im,1000,D,hamiltonian)
energy = TEBD.time_evolve_mpoham(ground,isingQuench,6,1000,D,hamiltonian)
plot(abs.(energy[:,1]), real.(energy[:,2]), show=true)
