using MPS
using TEBD
#using TensorOperations
using PyPlot
println("\n---quench.jl------------------------------------")

## parameters for the spin chain:
latticeSize = 10
maxBondDim = [10, 20, 40]
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
total_time_thermal = -im*[1.0]/2#, 1.0, 0.5, 0.1, 0.01]/2 # -im*total_time_thermal  for imag time evol
total_time_quench = 20.0
steps = 10000
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

## collect all quantities in on array for all betas to save into one file:
energy_all, entropy_all, magnetization_all, corr_fct_all, corr_length_all = [],[],[],[],[]



for beta_th in total_time_thermal
    println("beta_th = ", beta_th)

    for maxD in maxBondDim
        println("maxD = ", maxD)

        ## thermal state MPO:
        IDmpo = MPS.IdentityMPO(latticeSize,d)
        init_params = (J0, h0, g0)
        @time TEBD.time_evolve_mpoham(IDmpo,thermalIsing,beta_th,steps,maxD,0,init_params,ETH)
        # rho = MPS.multiplyMPOs(IDmpo,IDmpo) # --> IDmpo = exp[-beta/2 H]

        ## thermal quench:
        init_params = (J0, h0, g0)
        @time energy, entropy, magnetization, corr_fct, corr_length = TEBD.time_evolve_mpoham(IDmpo,isingQuench,total_time_quench,steps,maxD,0,init_params,ETH,"Isingthermal")


        ## Ising evolution:
        # init_params = (J0, h0, g0)
        # println("Norm: ", MPS.MPSnorm(mps_evol))
        # @time energy, entropy, magnetization, corr_fct, corr_length = TEBD.time_evolve_mpoham(mps_evol,isingQuench,total_time_quench,steps,maxD,entropy_cut,init_params,ETH,"Ising")
        # println("Norm: ", MPS.MPSnorm(mps_evol))
        # println( "E/N = ", MPS.mpoExpectation(mps_evol,hamiltonian)/(latticeSize-1) )


        ## Heisenberg evolution:
        # init_params = (Jx0, Jy0, Jz0, hx0)
        # println("Norm: ", MPS.MPSnorm(mps_evol))
        # @time energy, entropy, magnetization, corr_fct, corr_length = TEBD.time_evolve_mpoham(mps_evol,heisenbergQuench,total_time_quench,steps,maxD,entropy_cut,init_params,ETH,"Heisenberg")
        # println("Norm: ", MPS.MPSnorm(mps_evol))
        # println( "E/N = ", MPS.mpoExpectation(mps_evol,hamiltonian)/(latticeSize-1) )


        ## PLOTTING
        beta_plot = 2*real(im*beta_th)
        figure(1)
        plot(abs.(energy[:,1]), real.(energy[:,2]), label="\$\\beta_{th}\\, / \\,J = $beta_plot, D = $maxD\$")
        # figure(2)
        # plot(abs.(entropy[:,1]), real.(entropy[:,2]))
        figure(3)
        plot(abs.(magnetization[:,1]), real.(magnetization[:,2]))
        figure(4)
        plot(abs.(corr_fct[:,1]), real.(corr_fct[:,2]))
        # figure(5)
        # plot(abs.(corr_length[:,1]), real.(corr_length[:,2]))

        ## Collecting all data for saving
        if beta_th == total_time_thermal[1] && maxD == maxBondDim[1] # write out time only in first column
            energy_all = cat(2, abs.(energy[:,1]), real.(energy[:,2]))
            # entropy_all = cat(2, abs.(entropy[:,1]), real.(entropy[:,2]))
            magnetization_all = cat(2, abs.(magnetization[:,1]), real.(magnetization[:,2]))
            corr_fct_all = cat(2, abs.(corr_fct[:,1]), real.(corr_fct[:,2]))
            # corr_length_all = cat(2, abs.(corr_length[:,1]), real.(corr_length[:,2]))
        else
            energy_all = cat(2, energy_all, real.(energy[:,2]))
            # entropy_all = cat(2, entropy_all, real.(entropy[:,2]))
            magnetization_all = cat(2, magnetization_all, real.(magnetization[:,2]))
            corr_fct_all = cat(2, corr_fct_all, real.(corr_fct[:,2]))
            # corr_length_all = cat(2, corr_length_all, real.(corr_length[:,2]))
        end
    end
end

## PLOTTING and SAVING
figure(1)
xlabel("time")
ylabel("energy")
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
savefig("figures/energy.pdf")
writedlm("data/quench/energy.txt", energy_all)

# figure(2)
# xlabel("time")
# ylabel("entanglement entropy")
# savefig("figures/entanglement.pdf")
# writedlm("data/quench/entanglement.txt", entropy_all)

figure(3)
xlabel("time")
ylabel("magnetization")
savefig("figures/magnetization.pdf")
writedlm("data/quench/magnetization.txt", magnetization_all)

figure(4)
xlabel("time")
ylabel("correlation function")
savefig("figures/corr_fct.pdf")
writedlm("data/quench/corr_fct.txt", corr_fct_all)

# figure(5)
# xlabel("time")
# ylabel("correlation length")
# savefig("figures/corr_length.pdf")
# writedlm("data/quench/corr_length.txt", corr_length_all)



println("done: quench.jl")
show()
;
