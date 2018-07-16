using MPS
using TEBD
#using TensorOperations
using Plots
println("\n---quench.jl------------------------------------")

## parameters for the spin chain:
latticeSize = 30
maxBondDim = [20]
d = 2
prec = 1e-8

## Ising parameters:
J0 = 1.0
h0 = 1.0
g0 = 0.0

## TEBD parameters:
total_time_thermal = -im*[1/2]/2 # -im*total_time_thermal  for imag time evol
total_time_quench = 4.0
steps = 100

entropy_cut = Int(round(latticeSize/2)) # subsytem size for entanglement entopy; set to 0 to disregard

##operators to measure and quench parameters
J(time) = J0
h(time) = h0 + 0.1*exp(-20(time-0.5)^2)
g(time) = g0
opmag(time) = MPS.MpoFromOperators([[sx,Int(round(latticeSize/2))]],latticeSize)
opE(time) = MPS.IsingMPO(latticeSize,J(time),h(time),g(time))
quenchblocks(time) = TEBD.isingHamBlocks(latticeSize,J(time),h(time),g(time))
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
        @time TEBD.tebd_simplified(IDmpo,thermhamblocks,beta_th,steps,maxD,[])
        println("trace rho_th(0) = ", MPS.traceMPO(IDmpo,2))

        ## thermal quench:
        @time opvalues, err = TEBD.tebd_simplified(IDmpo,quenchblocks,total_time_quench,steps,maxD,operators)
        println("trace rho_th(t_max) = ", MPS.traceMPO(IDmpo,2))
        println(err)
        save_data(real.(opvalues),header=string(sth(latticeSize,2*real(beta_th),total_time_quench,steps,maxD),"t \t E \t mag\n"))
    end
end

println("done: quench.jl")
