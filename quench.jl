using MPS
using TEBD
#using TensorOperations
using Plots
println("\n---quench.jl------------------------------------")

## folder for saving:
subfolder = ""

## parameters for the spin chain:
latticeSize = 80
maxBondDim = [60]
d = 2
prec = 1e-10

## Ising parameters:
J0 = 1.0
h0 = 1.0
g0 = 0.0
q = 2*pi*(50/(latticeSize-1))
## TEBD parameters:
total_time_thermal = -im*[2]/2 # -im*total_time_thermal  for imag time evol
total_time_quench = 4.0
steps = 500
increment = 2 # stepsize > 1 after which physical quantities are calculated
entropy_cut = Int(round(latticeSize/2)) # subsytem size for entanglement entopy; set to 0 to disregard

##operators to measure and quench parameters
J(time) = ones(latticeSize)*J0
h(time) = ones(latticeSize)*h0 + sin.(q*(-1+(1:latticeSize)))*0.05*exp(-20(time-0.5)^2)
g(time) = ones(latticeSize)*g0
opmag(time) = MPS.MpoFromOperators([[sx,Int(round(latticeSize/2))]],latticeSize)
opE(time) = MPS.IsingMPO(latticeSize,J0,h0,g0)
quenchblocks(time) = TEBD.inhomogeneousIsingHamBlocks(latticeSize,J(time),h(time),g(time))
thermhamblocks(time) = TEBD.isingHamBlocks(latticeSize,J0,h0,g0)
operators = [opE opmag]
ham = MPS.IsingMPO(latticeSize,J0,h0,g0)

ETH = (false,0,0) # dummys: no ETH calcs here

function sth(L,beta,time,steps,D)
    return string("# L= ",latticeSize,"  beta= ",2*real(im*total_time_thermal),"  t_max= ",total_time_quench,"  steps= ",steps,",D= ", maxBondDim," #\n")
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

    for maxD in maxBondDim
        println("maxD = ", maxD)

        ## thermal state MPO:
        IDmpo = MPS.IdentityMPO(latticeSize,d)
        @time TEBD.tebd_simplified(IDmpo,thermhamblocks,beta_th,steps,maxD,[],prec)
        println("trace rho_th(0) = ", MPS.traceMPO(IDmpo,2))

        ## thermal quench:
        @time opvalues, err = TEBD.tebd_simplified(IDmpo,quenchblocks,total_time_quench,steps,maxD,operators,prec)
        println("trace rho_th(t_max) = ", MPS.traceMPO(IDmpo,2))
        println(err)
        save_data(real.(opvalues),header=string(sth(latticeSize,2*real(beta_th),total_time_quench,steps,maxD),"t \t E \t mag\n"))
    end
end


# ## PLOTTING and SAVING
# figure(1)
# xlabel("\$t\\, /\\, J \$")
# ylabel("\$E(t)\$")
# title("energy")
# legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
# savefig("figures/"*subfolder*"/energy.pdf")
# open("data/quench/"*subfolder*"/energy.txt", "w") do f
#     if length(total_time_thermal) > 1
#         write(f, string("# L= ",latticeSize,"  D= ",maxBondDim,"  t_max= ",total_time_quench,"  steps= ",steps,"\n"))
#         write(f, string("t \t  beta= ", 2*real(im*total_time_thermal), "\n"))
#     else
#         write(f, string("# L= ",latticeSize,"  beta= ",2*real(im*total_time_thermal),"  t_max= ",total_time_quench,"  steps= ",steps,"\n"))
#         write(f, string("t \t  D= ", maxBondDim, "\n"))
#     end
# end
# open("data/quench/"*subfolder*"/energy.txt", "a") do f
#     writedlm(f, energy_all)
# end
#
# # figure(2)
# # xlabel("time")
# # ylabel("entanglement entropy")
# # savefig("figures/entanglement.pdf")
# # writedlm("data/quench/entanglement.txt", entropy_all)
#
# figure(3)
# xlabel("\$t\\, /\\, J \$")
# ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
# title("magnetization")
# savefig("figures/"*subfolder*"/magnetization.pdf")
# open("data/quench/"*subfolder*"/magnetization.txt", "w") do f
#     if length(total_time_thermal) > 1
#         write(f, string("# L= ",latticeSize,"  D= ",maxBondDim,"  t_max= ",total_time_quench,"  steps= ",steps,"\n"))
#         write(f, string("t \t  beta= ", 2*real(im*total_time_thermal), "\n"))
#     else
#         write(f, string("# L= ",latticeSize,"  beta= ",2*real(im*total_time_thermal),"  t_max= ",total_time_quench,"  steps= ",steps,"\n"))
#         write(f, string("t \t  D= ", maxBondDim, "\n"))
#     end
# end
# open("data/quench/"*subfolder*"/magnetization.txt", "a") do f
#     writedlm(f, magnetization_all)
# end
#
#
# figure(4)
# xlabel("\$t\\, /\\, J \$")
# ylabel("\$\\langle \\sigma_z(L/4) \\, \\sigma_z(3/4 L) \\rangle\$")
# title("correlation function")
# savefig("figures/"*subfolder*"/corr_fct.pdf")
# open("data/quench/"*subfolder*"/corr_fct.txt", "w") do f
#     if length(total_time_thermal) > 1
#         write(f, string("# L= ",latticeSize,"  D= ",maxBondDim,"  t_max= ",total_time_quench,"  steps= ",steps,"\n"))
#         write(f, string("t \t  beta= ", 2*real(im*total_time_thermal), "\n"))
#     else
#         write(f, string("# L= ",latticeSize,"  beta= ",2*real(im*total_time_thermal),"  t_max= ",total_time_quench,"  steps= ",steps,"\n"))
#         write(f, string("t \t  D= ", maxBondDim, "\n"))
#     end
# end
# open("data/quench/"*subfolder*"/corr_fct.txt", "a") do f
#     writedlm(f, corr_fct_all)
# end
#
# # figure(5)
# # xlabel("time")
# # ylabel("correlation length")
# # savefig("figures/corr_length.pdf")
# # writedlm("data/quench/corr_length.txt", corr_length_all)
#
#
#
#
# println("done: quench.jl")
