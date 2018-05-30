using MPS
using TEBD
#using TensorOperations
using PyPlot
println("\n---quench.jl------------------------------------")

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
total_time_thermal = -im*4.0    # -im*total_time_thermal  for imag time evol
total_time_quench = 10.0
steps = 500
entropy_cut = Int(round(latticeSize/2)) # subsytem size for entanglement entopy; set to 0 to disregard

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

# hamiltonian = MPS.IsingMPO(latticeSize, J0, h0, g0)
# hamiltonian = MPS.HeisenbergMPO(latticeSize, Jx0, Jy0, Jz0, hx0)

# mps = MPS.randomMPS(latticeSize,d,maxBondDim)
# MPS.makeCanonical(mps)
# ground,Eground = MPS.DMRG(mps,hamiltonian,prec)
# mps_evol = mps # mps used in TEBD (e.g. ground in quench or mps for imag time evolution)
ETH = (false,0,0) # dummys: no ETH calcs here




## thermal state MPO:
IDmpo = MPS.IdentityMPO(latticeSize,d)
init_params = (J0, h0, g0)
@time TEBD.time_evolve_mpoham(IDmpo,thermalIsing,total_time_thermal,steps,maxBondDim,0,init_params,ETH)
# rho = MPS.multiplyMPOs(IDmpo,IDmpo) # --> IDmpo = exp[-beta/2 H]


## Ising evolution:
# init_params = (J0, h0, g0)
# println("Norm: ", MPS.MPSnorm(mps_evol))
# @time energy, entropy, magnetization = TEBD.time_evolve_mpoham(mps_evol,isingQuench,total_time_quench,steps,maxBondDim,entropy_cut,init_params,ETH,"Ising")
# println("Norm: ", MPS.MPSnorm(mps_evol))
# println( "E/N = ", MPS.mpoExpectation(mps_evol,hamiltonian)/(latticeSize-1) )

## thermal quench:
init_params = (J0, h0, g0)
@time energy, entropy, magnetization = TEBD.time_evolve_mpoham(IDmpo,isingQuench,total_time_quench,steps,maxBondDim,0,init_params,ETH,"Isingthermal")


## Heisenberg evolution:
# init_params = (Jx0, Jy0, Jz0, hx0)
# println("Norm: ", MPS.MPSnorm(mps_evol))
# @time energy, entropy, magnetization = TEBD.time_evolve_mpoham(mps_evol,heisenbergQuench,total_time_quench,steps,maxBondDim,entropy_cut,init_params,ETH,"Heisenberg")
# println("Norm: ", MPS.MPSnorm(mps_evol))
# println( "E/N = ", MPS.mpoExpectation(mps_evol,hamiltonian)/(latticeSize-1) )


## PLOTTING
figure(1)
plot(abs.(energy[:,1]), real.(energy[:,2]))
xlabel("time")
ylabel("energy")

# figure(2)
# plot(abs.(entropy[:,1]), real.(entropy[:,2]))
# xlabel("time")
# ylabel("entanglement entropy")

figure(3)
plot(abs.(magnetization[:,1]), real.(magnetization[:,2]))
xlabel("time")
ylabel("magnetization")



show()
;
