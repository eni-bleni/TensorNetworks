#---------------------------------------------------------------#
###   check whether final state after quench is thermalized   ###
#---------------------------------------------------------------#
using MPS
using TEBD
# using layout
using PyPlot
println("\n---thermalization.jl------------------------------------")

## data folder:
subfolder = "Instantaneous_Tstudies"

## parameters for the spin chain:
latticeSize = 50
maxBondDim = 200
d = 2
prec = 1e-20

## Ising parameters:
J0 = 1.0
h0 = 1.0+0.1 #-0.525
g0 = 0.0 # 0.25

## TEBD parameters:
steps = Int(100e3)

hamiltonian = MPS.IsingMPO(latticeSize, J0, h0, g0)
thermhamblocks(t) = TEBD.isingHamBlocks(latticeSize,J0,h0,g0)

## thermalization time period:
t_min = 1.5
t_max = 4.0


function save_data(data, filename= string(@__DIR__,"/data/quench/"*subfolder*"/thermalization.txt"); header="")
    open(filename, "a") do f
        write(f, header)
        writedlm(f, data)
    end
end
# write header:
save_data([], header=string("# beta_th \t E_final \t E_thermal \t beta_final \t sx_final \n"))


## read out data:
f = open("data/quench/"*subfolder*"/opvalues.txt")
lines = readlines(f)
close(f)
sep_inds = findin(lines, [""])

for i = 1:length(sep_inds)
    counter = 1

    if i==1
        num_steps = parse(split(lines[1])[8])
        E_mps = Array{Float64}(num_steps, 2)
        magnetization_mps = Array{Float64}(num_steps, 2)
        L = include_string(split(lines[1])[2])
        beta_th = include_string(split(lines[1])[4])
        for l = 3 : sep_inds[1]-1
            line = parse.(split(lines[l]))
            E_mps[counter,:] = [line[1] line[2]]
            magnetization_mps[counter,:] = [line[1] line[3]]
            counter += 1
        end
    else
        num_steps = parse(split(lines[sep_inds[i-1]+1])[8])
        E_mps = Array{Float64}(num_steps, 2)
        magnetization_mps = Array{Float64}(num_steps, 2)
        L = include_string(split(lines[sep_inds[i-1]+1])[2])
        beta_th = include_string(split(lines[sep_inds[i-1]+1])[4])
        for l = sep_inds[i-1]+3 : sep_inds[i]-1
            line = parse.(split(lines[l]))
            E_mps[counter,:] = [line[1] line[2]]
            magnetization_mps[counter,:] = [line[1] line[3]]
            counter += 1
        end
    end

    time = Float64.(magnetization_mps[:,1])
    E_t = Float64.(E_mps[:,2])
    sx_t = Float64.(magnetization_mps[:,2])

    ind_min = maximum(find(time .<= t_min))
    ind_max = minimum(find(time .>= t_max))

    # find corresponding thermal state with associated physical quantities:
    E_final = mean(E_t[ind_min:ind_max]) # asymptotic const energy value
    ETH = (true,E_final,hamiltonian)
    total_time_thermal = E_final <= E_t[1] ? -im*beta_th/2*1.4 : -im*beta_th/2
    IDmpo = MPS.IdentityMPO(latticeSize,d)
    E_thermal, betahalf = TEBD.tebd_simplified(IDmpo,thermhamblocks,total_time_thermal,steps,maxBondDim,[], ETH, tol=prec)
    opmag = MPS.MpoFromOperators([[sx,Int(round(latticeSize/2))]],latticeSize)
    sx_final = real(MPS.traceMPOprod(IDmpo,opmag,2))

    # Outputs:
    save_data([beta_th E_final E_thermal 2*betahalf sx_final])

    println("\nbeta_th = ", beta_th)
    println("E_thermal, beta/2 = ", E_thermal, ", ", betahalf)
    println("E_final, diff = ", E_final, ", ", abs(E_final-E_thermal))
    println("sx_final = ", sx_final)

    figure(1)
    plot(time,E_t)
    axhline(E_final, ls="--",c="k")
    axhline(E_thermal, ls="--",c="b")

    figure(2)
    plot(time, sx_t)
    axhline(sx_final, ls="--",c="b")
end



## PLOTS
figure(1)
xlabel("\$t\\, /\\, J \$")
ylabel("\$E(t)\$")
title("\$energy\$")
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
savefig("figures/"*subfolder*"/energy_thermalized.pdf")

figure(2)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
title("\$magnetization\$")
savefig("figures/"*subfolder*"/magnetization_thermalized.pdf")











# #######
show()
;
