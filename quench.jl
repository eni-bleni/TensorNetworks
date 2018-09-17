using MPS
using TEBD
#using TensorOperations
using PyPlot
println("\n---quench.jl------------------------------------")

## folder for saving figures:
subfolder = ""

## parameters for the spin chain:
latticeSize = 50
maxBondDim = [300]
d = 2
prec = 1e-20 # for TEBD
prec_DMRG = 1e-12

## Ising parameters:
J0 = 1.0
h0 = -0.525
g0 = 0.25

## TEBD parameters:
total_time_thermal = -im*[0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0]/2 # -im*total_time_thermal  for imag time evol
total_time_quench = 10.0
steps_th = 1000 # no of steps to construct thermal state
steps = 2000    # no of steps in real time evolution
step_incr = 10  # specifies after how many steps phys quantities are calculated
entropy_cut = Int(round(latticeSize/2)) # subsytem size for entanglement entopy; set to 0 to disregard

## what to do:
analyze_thermal_states = false  # calculate correlation fct for all beta_th
analyze_correlations = false     # correlator spreading: calculate <sz sz> for all distances during quench
perform_th_quench = true       # thermal quench
perform_gs_quench = false       # ground state quench

## quench parameters:
J(time) = J0
delta = 0.1
# h(time) = h0 + exp(-3(time-2)^2)                 # large Gaussian quench
h(time) = h0 + delta*exp(-20(time-0.5)^2)          # small Gaussian quench
# h(time) = time < 0.5 ? h0 : h0+delta             # instantaneous quench
# h(time) = h0 + delta*(1 + tanh(10*(time-0.5)))/2 # continuous quench
g(time) = g0

## operators to measure:
opE(time) = MPS.IsingMPO(latticeSize,J(time),h(time),g(time))
opmag(time) = MPS.MpoFromOperators([[sx,Int(round(latticeSize/2))]],latticeSize)
spin_pos = [[sz,Int(floor(latticeSize/4))], [sz,Int(floor(3/4*latticeSize))]] # position of spins in chain for correlation fct
opcorr(time) = MPS.MpoFromOperators(spin_pos,latticeSize)
operators = [opE opmag opcorr]

## Hamiltonians:
quenchblocks(time) = TEBD.isingHamBlocks(latticeSize,J(time),h(time),g(time))
thermhamblocks(time) = TEBD.isingHamBlocks(latticeSize,J0,h0,g0)
ham0 = MPS.IsingMPO(latticeSize,J0,h0,g0)

## Variables:
opvalues = []
err = []
entropy = []
ETH = (false,0,0) # dummys: no ETH calcs here
incr = 1 # space distance increment for correlator spreading



function sth(L,beta,time,steps,D)
    return string("L= ",L,"  beta= ",beta,"  t_max= ",time,"  steps= ",steps,"  D= ", D,"  prec= ",prec,"  delta= ",delta,"\n")
end

function sth2(L,beta,steps,D)
    return string("L= ",L,"  beta= ",beta,"  steps= ",steps,"  D= ", D,"  prec= ",prec,"\n")
end

function save_data(data, filename= string(@__DIR__,"/data/quench/opvalues.txt"); header="")
    open(filename, "a") do f
        write(f, header)
        writedlm(f, data)
        write(f,"\r\n")
    end
end


## ZERO temperature
if perform_gs_quench
    for maxD in maxBondDim
        println("\n...performing ground state DMRG...")
        mps = MPS.randomMPS(latticeSize,d,maxD)
        MPS.makeCanonical(mps)

        states,energies = MPS.n_lowest_states(mps, ham0, prec_DMRG, 1)
        ground,E0 = states[1], energies[1]
        println("E0/L = ", E0/(latticeSize-1))

        println("\n...performing gs quench...")
        @time opvalues, err, entropy = TEBD.tebd_simplified(ground,quenchblocks,total_time_quench,steps,maxD,operators, ETH, entropy_cut, increment=step_incr, tol=prec)

        ## Plotting:
        opvalues = real.(opvalues)
        figure(1)
        plot(opvalues[:,1], opvalues[:,2])
        figure(3)
        plot(opvalues[:,1], opvalues[:,3])
        figure(4)
        plot(opvalues[:,1], opvalues[:,4])
        figure(6)
        plot(opvalues[:,1], err)
        figure(7)
        plot(opvalues[:,1], entropy)

        save_data(cat(2,opvalues,err,entropy), header=string(sth(latticeSize,0.0,total_time_quench,steps,maxD), "# t \t E \t mag \t corr \t err \t entropy\n"))
    end
end


## FINITE temperature
if analyze_thermal_states || perform_th_quench || analyze_correlations
    for beta_th in total_time_thermal
        println("\nbeta_th = ", 2*real(im*beta_th))
        beta_plot = 2*real(im*beta_th)

        for maxD in maxBondDim
            println("maxD = ", maxD)

            ## thermal state MPO:
            IDmpo = MPS.IdentityMPO(latticeSize,d)
            @time TEBD.tebd_simplified(IDmpo,thermhamblocks,beta_th,steps_th,maxD,[], ETH, tol=prec)
            println("trace rho_th(0) = ", MPS.traceMPO(IDmpo,2))

            if analyze_thermal_states
                ## correlation functions:
                corr = []
                for m = 1:incr:latticeSize-1
                    spin_pos = [[sz,1], [sz,1+m]]
                    corr_m = MPS.traceMPOprod(IDmpo,MPS.MpoFromOperators(spin_pos,latticeSize),2) - MPS.traceMPOprod(IDmpo,MPS.MpoFromOperators([spin_pos[1]],latticeSize),2)*MPS.traceMPOprod(IDmpo,MPS.MpoFromOperators([spin_pos[2]],latticeSize),2)
                    push!(corr, corr_m)
                    println("<sz_1 sz_",1+m,"> = ", corr_m)
                end

                figure(7)
                plot(1:incr:latticeSize-1, abs.(corr), ls="", marker="s", label="\$\\beta_{th}\\, / \\,J = $beta_plot, D = $maxD\$")

                save_data(cat(2,collect(1:incr:latticeSize-1),real.(corr)), string(@__DIR__,"/data/correlations/corr_fcts.txt"); header=string(sth2(latticeSize,2*real(im*beta_th),steps,maxD), "# m \t <sz_1 sz_{1+m}>\n"))
            end

            if perform_th_quench
                ## thermal quench:
                @time opvalues, err = TEBD.tebd_simplified(IDmpo,quenchblocks,total_time_quench,steps,maxD,operators, ETH, increment=step_incr, tol=prec)
                println("trace rho_th(t_max) = ", MPS.traceMPO(IDmpo,2))

                ## Plotting:
                opvalues = real.(opvalues)
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

            if analyze_correlations
                ## thermal quench:
                Lhalf = Int(floor(latticeSize/2))
                dist_interval = cat(1,-Lhalf+1:incr:-1, 1:incr:Lhalf-1)
                corr_params = ("corr_fct", Lhalf, dist_interval)
                @time opvalues, err = TEBD.tebd_simplified(IDmpo,quenchblocks,total_time_quench,steps,maxD,corr_params, ETH, increment=step_incr, tol=prec)
                println("trace rho_th(t_max) = ", MPS.traceMPO(IDmpo,2))

                save_data(real.(opvalues), string(@__DIR__,"/data/quench/corr_spreading.txt"); header=string(sth(latticeSize,beta_plot,total_time_quench,steps,maxD), "# t \t <sz_{L/2} sz_{L/2+m}> \t m= ",dist_interval,"\n"))
            end
        end
    end
end


## PLOTTING and SAVING
if perform_th_quench || perform_gs_quench
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
end

if analyze_thermal_states
    figure(7)
    xlabel("\$ m \$")
    ylabel("\$\\vert \\langle\\sigma_z(1) \\, \\sigma_z(1+m)\\rangle - \\langle\\sigma_z(1)\\rangle \\langle\\sigma_z(1+m)\\rangle \\vert\$")
    title("correlation function")
    legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
    savefig("figures/"*subfolder*"/corr_fct_distance.pdf")
end

if perform_gs_quench
    figure(7)
    xlabel("\$t\\, /\\, J \$")
    ylabel("\$S_{L/2}\$")
    title("entanglement entropy")
    savefig("figures/"*subfolder*"/entropy.pdf")
end






println("done: quench.jl")
show()
;
