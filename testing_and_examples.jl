using MPS
using TEBD
# using Plots
using PyPlot
println("\n---------------------------------------")


## parameters for the spin chain:
latticeSize = 100
maxBondDim = 100
d = 2
prec = 1e-8

## Ising parameters:
J = 1.0
h = 1.0
g = 0.0

## Heisenberg parameters:
Jx = 1.0
Jy = 1.0
Jz = 1.0
hx = 1.0

## TEBD parameters:
total_time = -im*10.0 # -im*total_time  for imag time evol
steps = 1000

# define Pauli matrices:
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]



################################################################################
##                                  DMRG
################################################################################
# println("...performing ground state DMRG...")
#
# hamiltonian = MPS.IsingMPO(latticeSize, J, h, g)
# # hamiltonian = MPS.HeisenbergMPO(latticeSize, Jx, Jy, Jz, hx)
#
# mps = MPS.randomMPS(latticeSize,d,maxBondDim)
# MPS.makeCanonical(mps)
#
# @time ground,E0 = MPS.DMRG(mps,hamiltonian,prec)
# println("E/N = ", E0/(latticeSize-1))
#
#
# println("\n...performing excited state DMRG...")
# @time exc,E = MPS.DMRG(mps,hamiltonian,prec,ground)
# println("Overlap: ",MPS.MPSoverlap(exc,ground))


################################################################################
##                           Entanglement Entropy
################################################################################
# println("\n...entanglement entropy...")
#
# entropy = Array{Any}(latticeSize,2)
# for i = 0:latticeSize-1
#     entropy[i+1,1] = i+1 # subsystem size
#     entropy[i+1,2] = MPS.entropy(ground,i)
# end
# ind_min = 5
# ind_max = Int(round(0.3*latticeSize))
# fit_interval = log.(entropy[ind_min:ind_max,1])
# a,b = linreg(fit_interval, entropy[ind_min:ind_max,2])
# c = 6*b
# println("central charge (if critical): ", c)
#
# figure(1)
# plot(entropy[:,1], entropy[:,2])
# xlabel("\$L\$ (subsystem size)")
# ylabel("\$S_L\$ (entanglement entropy)")
#
# figure(2)
# plot(fit_interval, a+b*fit_interval)
# plot(fit_interval, entropy[ind_min:ind_max,2])
# xlabel("\$\\ln(L)\$")
# ylabel("\$S_L\$")
# # text(fit_interval[end]*0.6, entropy[ind_max,2]*0.99, "c = %1.2f" %c)


################################################################################
##                           Correlation Length
################################################################################
println("\n...correlation length...")

mps = MPS.randomMPS(latticeSize,d,maxBondDim)
MPS.makeCanonical(mps)

corr, xi, ind_max, a, b = MPS.correlation_length(mps, d)

figure(3)
semilogy(corr[1:end,1], abs.(corr[1:end,2]))
plot(corr[1:ind_max,1], exp.(a+b*corr[1:ind_max,1]))
xlabel("distance \$m\$")
ylabel("\$\\vert\\langle O_1^{(1)}O_2^{(m)}\\rangle - \\langle O_1^{(1)}\\rangle\\langle O_2^{(m)}\\rangle\\vert\$")



################################################################################
##                                  TEBD
################################################################################

### ATTENTION: uses not correct hamiltonian at endpoints, use quench.jl instead

# println("\n...performing TEBD...")
#
# # ham = TEBD.TwoSiteIsingHamiltonian(J,h,g)
# ham = TEBD.TwoSiteHeisenbergHamiltonian(Jx,Jy,Jz,hx)
#
# mps2 = MPS.randomMPS(latticeSize,d,maxBondDim)
# # mps2 = ground
# MPS.makeCanonical(mps2)
#
# println("Norm: ", MPS.MPSnorm(mps2)," Energy: ", MPS.mpoExpectation(mps2,hamiltonian))
#
# magnetization = MPS.OneSiteMPO(latticeSize, Int(round(latticeSize/2)), sx)
# expect = TEBD.time_evolve(mps2, ham, total_time, steps, maxBondDim, hamiltonian)
#
# println("Norm: ", MPS.MPSnorm(mps2)," Energy: ", MPS.mpoExpectation(mps2,hamiltonian))
# println( "E/N = ", MPS.mpoExpectation(mps2,hamiltonian)/(latticeSize-1) )
#
#
# ## PLOTTING
# plot(abs.(expect[:,1]), real.(expect[:,2]), show=true)

show()
;
