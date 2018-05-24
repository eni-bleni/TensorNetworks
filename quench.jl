using MPS
using TEBD
#using TensorOperations
using PyPlot
println("\n---------------------------------------")

## parameters for the spin chain:
latticeSize = 10
maxBondDim = 20
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
total_time = -im*10.0    # -im*total_time  for imag time evol
steps = 1000
entropy_cut = Int(round(latticeSize/2)) # subsytem size for entanglement entopy; set to 0 to disregard

# define Pauli matrices:
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]



function isingQuench(i, time, params)
    J0, h0, g0 = params
    ZZ = kron(sz, sz)
    ZI = kron(sz, si)
    IZ = kron(si, sz)
    XI = kron(sx, si)
    IX = kron(si, sx)

    J, h, g = TEBD.evolveIsingParams(J0, h0, g0, time)

    if i==1
        return J*ZZ + h/2*(XI+2IX) + g/2*(ZI+2IZ)
    elseif i==latticeSize-1
        return J*ZZ + h/2*(2*XI+IX) + g/2*(2*ZI+IZ)
    else
        return J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ)
    end
end

function heisenbergQuench(i,time, params)
    Jx0, Jy0, Jz0, hx0 = params
    XX = kron(sx, sx)
    YY = kron(sy, sy)
    ZZ = kron(sz, sz)
    XI = kron(sx, si)
    IX = kron(si, sx)

    Jx, Jy, Jz, hx = TEBD.evolveHeisenbergParams(Jx0, Jy0, Jz0, hx0, time)

    if i==1
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+2*IX)
    elseif i==latticeSize-1
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(2*XI+IX)
    else
        return Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+IX)
    end
end

hamiltonian = MPS.IsingMPO(latticeSize, J0, h0, g0)
# hamiltonian = MPS.HeisenbergMPO(latticeSize, Jx0, Jy0, Jz0, hx0)

mps = MPS.randomMPS(latticeSize,d,maxBondDim)
MPS.makeCanonical(mps)
ground,Eground = MPS.DMRG(mps,hamiltonian,prec)
mps_evol = mps # mps used in TEBD (e.g. ground in quench or mps for imag time evolution)
ETH = (false,0,0) # dummys: no ETH calcs here

# magnetMPO = MPS.OneSiteMPO(latticeSize, Int(round(latticeSize/2)), [0 1; 1 0])
# magnetization = TEBD.time_evolve(mps2, ham, total_time, steps, maxBondDim, magnetMPO)
## PLOTTING
# plot(abs.(expect[:,1]), real.(expect[:,2]), show=true)



## thermal state MPO:
init_params = (J0, h0, g0)
IDmpo = MPS.IdentityMPO(latticeSize,d)
@time TEBD.time_evolve(IDmpo,isingQuench,total_time,steps,maxBondDim,0,init_params,ETH)
rho = MPS.multiplyMPOs(IDmpo,IDmpo)


## Ising evolution:
# init_params = (J0, h0, g0)
# println("Norm: ", MPS.MPSnorm(mps_evol))
# @time energy, entropy = TEBD.time_evolve(mps_evol,isingQuench,total_time,steps,maxBondDim,entropy_cut,init_params,ETH,"Ising")
# println("Norm: ", MPS.MPSnorm(mps_evol))
# println( "E/N = ", MPS.mpoExpectation(mps_evol,hamiltonian)/(latticeSize-1) )


## Heisenberg evolution:
# init_params = (Jx0, Jy0, Jz0, hx0)
# println("Norm: ", MPS.MPSnorm(mps_evol))
# @time energy, entropy = TEBD.time_evolve(mps_evol,heisenbergQuench,total_time,steps,maxBondDim,entropy_cut,init_params,ETH,"Heisenberg")
# println("Norm: ", MPS.MPSnorm(mps_evol))
# println( "E/N = ", MPS.mpoExpectation(mps_evol,hamiltonian)/(latticeSize-1) )


## PLOTTING
# figure(1)
# plot(abs.(energy[:,1]), real.(energy[:,2]))
# xlabel("time")
# ylabel("energy")
#
# figure(2)
# plot(abs.(entropy[:,1]), real.(entropy[:,2]))
# xlabel("time")
# ylabel("entanglement entropy")


show()
;
