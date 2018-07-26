using MPS
using TEBD
#using TensorOperations
using PyPlot
println("\n---quench.jl------------------------------------")

## folder for saving figures:
subfolder = ""

## parameters for the spin chain:
latticeSize = 50
maxBondDim = [200]
d = 2
prec = 1e-20

## Ising parameters:
J0 = 1.0
h0 = 1.0
g0 = 0.0

## TEBD parameters:
total_time_thermal = -im*[0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0]/2 # -im*total_time_thermal  for imag time evol
total_time_quench = 5.0
steps = 1000
analyze_thermal_states = false # calculate only correlation fct for several beta_th
increment = 2 # stepsize > 1 after which physical quantities are calculated
entropy_cut = Int(round(latticeSize/2)) # subsytem size for entanglement entopy; set to 0 to disregard

## operators to measure and quench parameters:
J(time) = J0
# h(time) = h0 + exp(-3(time-2)^2)
# h(time) = h0 + 0.1*exp(-20(time-0.5)^2) # Gaussian quench
h(time) = time < 0.5 ? h0 : h0+0.1      # instantaneous quench
g(time) = g0

opE(time) = MPS.IsingMPO(latticeSize,J(time),h(time),g(time))
opmag(time) = MPS.MpoFromOperators([[sx,Int(round(latticeSize/2))]],latticeSize)
spin_pos = [[sz,Int(floor(latticeSize/4))], [sz,Int(floor(3/4*latticeSize))]] # position of spins in chain for correlation fct
opcorr(time) = MPS.MpoFromOperators(spin_pos,latticeSize)
operators = [opE opmag opcorr]

quenchblocks(time) = TEBD.isingHamBlocks(latticeSize,J(time),h(time),g(time))
thermhamblocks(time) = TEBD.isingHamBlocks(latticeSize,J0,h0,g0)
ham = MPS.IsingMPO(latticeSize,J0,h0,g0)
opvalues = []
err = []
ETH = (false,0,0) # dummys: no ETH calcs here



function sth(L,beta,time,steps,D)
    return string("L= ",L,"  beta= ",beta,"  t_max= ",time,"  steps= ",steps,"  D= ", D,"  prec= ",prec,"\n")
end

function save_data(data, filename= string(@__DIR__,"/data/quench/opvalues.txt"); header="")
    open(filename, "a") do f
        write(f, header)
        writedlm(f, data)
        write(f,"\r\n")
    end
end

for beta_th in total_time_thermal
    println("beta_th = ", 2*real(im*beta_th))

    if analyze_thermal_states
        println("to do")
    else
        for maxD in maxBondDim
            println("maxD = ", maxD)

            ## thermal state MPO:
            IDmpo = MPS.IdentityMPO(latticeSize,d)
            @time TEBD.tebd_simplified(IDmpo,thermhamblocks,beta_th,steps,maxD,[], tol=prec)
            println("trace rho_th(0) = ", MPS.traceMPO(IDmpo,2))

            ## thermal quench:
            @time opvalues, err = TEBD.tebd_simplified(IDmpo,quenchblocks,total_time_quench,steps,maxD,operators, tol=prec)
            println("trace rho_th(t_max) = ", MPS.traceMPO(IDmpo,2))

            ## Plotting:
            opvalues = real.(opvalues)
            beta_plot = 2*real(im*beta_th)
            figure(1)
            plot(opvalues[:,1], opvalues[:,2], label="\$\\beta_{th}\\, / \\,J = $beta_plot, D = $maxD\$")
            figure(3)
            plot(opvalues[:,1], opvalues[:,3])
            figure(4)
            plot(opvalues[:,1], opvalues[:,4])
            figure(6)
            plot(opvalues[:,1], err)

            save_data(cat(2,opvalues,err), header=string(sth(latticeSize,beta_plot,total_time_quench,steps,maxD), "# t \t E \t mag \t corr \t err\n"))
        end
    end
end


## PLOTTING and SAVING
figure(1)
xlabel("\$t\\, /\\, J \$")
ylabel("\$E(t)\$")
title("energy")
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
savefig("figures/"*subfolder*"/energy.pdf")

# figure(2)
# xlabel("time")
# ylabel("entanglement entropy")
# savefig("figures/entanglement.pdf")
# writedlm("data/quench/entanglement.txt", entropy_all)

figure(3)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
title("magnetization")
savefig("figures/"*subfolder*"/magnetization.pdf")

figure(4)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_z(L/4) \\, \\sigma_z(3/4 L) \\rangle\$")
title("correlation function")
savefig("figures/"*subfolder*"/corr_fct.pdf")

# figure(5)
# xlabel("time")
# ylabel("correlation length")
# savefig("figures/corr_length.pdf")
# writedlm("data/quench/corr_length.txt", corr_length_all)

figure(6)
xlabel("\$t\\, /\\, J \$")
title("error")
savefig("figures/"*subfolder*"/error.pdf")









println("done: quench.jl")
show()
;
