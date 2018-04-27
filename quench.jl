using MPS
using TEBD
#using TensorOperations
using Plots
println("\n---------------------------------------")

prec = 1e-8
D = 40
L = 10
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]
# J=1
# h0 = 1

function isingQuench(i,time)
    ZZ = kron(sz, sz)
    ZI = kron(sz, si)
    IZ = kron(si, sz)
    XI = kron(sx, si)
    IX = kron(si, sx)
    h = h0 #+ 5*exp(-10(time-2)^2) #sin(time) *time^2*(pi - time)^2
    g = 0
    if i==1
        return J*ZZ + h/2*(XI+2IX) + g/2*(ZI+2IZ)
    elseif i==L-1
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

    Jx = 1.0
    Jy = 1.0
    Jz = 1.0
    hx0 = 1.0
    hx = hx0 #+ 0*5*exp(-10(time-2)^2) #sin(time) *time^2*(pi - time)^2

    if i==1
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+2*IX)
    elseif i==L-1
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(2*XI+IX)
    else
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+IX)
    end
end

#hamiltonian = MPS.IsingMPO(L,J,h0,0)
Jx = 1.0
Jy = 1.0
Jz = 1.0
hx = 1.0
hamiltonian = MPS.HeisenbergMPO(L, Jx, Jy, Jz, hx)


mps = MPS.randomMPS(L,2,D)
MPS.makeCanonical(mps)
ground,Eground = MPS.DMRG(mps,hamiltonian,prec)


# magnetMPO = MPS.OneSiteMPO(latticeSize, Int(round(latticeSize/2)), [0 1; 1 0])
# magnetization = TEBD.time_evolve(mps2, ham, total_time, steps, maxBondDim, magnetMPO)

## PLOTTING
# plot(abs.(expect[:,1]), real.(expect[:,2]), show=true)

# energy = TEBD.time_evolve_mpoham(ground,isingQuench,-10im,1000,D,hamiltonian)
energy = TEBD.time_evolve_mpoham(ground,heisenbergQuench,-5im,1000,D,hamiltonian)
plot(abs.(energy[:,1]), real.(energy[:,2]), show=true)
