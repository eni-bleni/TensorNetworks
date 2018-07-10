using layout
using PyPlot


### read out data:

subfolder = "1e-1shortdetailGauss_L50_beta0.5_J1_h1"

E_mps, header = readdlm("data/quench/"*subfolder*"/energy.txt", header=true)
magnetization_mps, header = readdlm("data/quench/"*subfolder*"/magnetization.txt", header=true)
corr_fct_mps, header = readdlm("data/quench/"*subfolder*"/corr_fct.txt", header=true)
bondDims = include_string(join(header[3:end]))

plot_comparison = false # control var for comparison to exact numerics (small L)
if "energy_exact.txt" in readdir("data/quench/"*subfolder)
    E_exact = readdlm("data/quench/"*subfolder*"/energy_exact.txt")
    magnetization_exact = readdlm("data/quench/"*subfolder*"/magnetization_exact.txt")
    corr_fct_exact = readdlm("data/quench/"*subfolder*"/corr_fct_exact.txt")
    plot_comparison = true
end



### convergence with increasing D:

for i = 1:length(bondDims)
    D = bondDims[i]

    figure(1)
    plot(E_mps[:,1], E_mps[:,i+1], label="\$D = $D\$")
    figure(2)
    plot(magnetization_mps[:,1], magnetization_mps[:,i+1])
    figure(3)
    plot(corr_fct_mps[:,1], corr_fct_mps[:,i+1])
end

figure(1)
xlabel("\$t\\, /\\, J \$")
ylabel("\$E(t)\$")
title("\$energy\$")
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
layout.nice_ticks()
savefig("figures/"*subfolder*"/energy.pdf")

figure(2)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
title("\$magnetization\$")
layout.nice_ticks()
savefig("figures/"*subfolder*"/magnetization.pdf")

figure(3)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_z(L/4) \\, \\sigma_z(3/4 L) \\rangle\$")
title("\$correlation\\, function\$")
layout.nice_ticks()
savefig("figures/"*subfolder*"/corr_fct.pdf")



### compare exact numerical time evolution with MPS result (for highest D):

if plot_comparison
    figure(4)
    plot(E_mps[:,1], E_mps[:,length(bondDims)+1], label="MPS", c="k")
    plot(E_exact[:,1], E_exact[:,2], ls="--", label="exact", c="orange")
    xlabel("\$t\\, /\\, J \$")
    ylabel("\$E(t)\$")
    title("\$energy\$")
    legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
    layout.nice_ticks()
    savefig("figures/"*subfolder*"/energy_comp.pdf")

    figure(5)
    plot(magnetization_mps[:,1], magnetization_mps[:,length(bondDims)+1], c="k")
    plot(magnetization_exact[:,1], magnetization_exact[:,2], ls="--", c="orange")
    xlabel("\$t\\, /\\, J \$")
    ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
    title("\$magnetization\$")
    layout.nice_ticks()
    savefig("figures/"*subfolder*"/magnetization_comp.pdf")

    figure(6)
    plot(corr_fct_mps[:,1], corr_fct_mps[:,length(bondDims)+1], c="k")
    plot(corr_fct_exact[:,1], corr_fct_exact[:,2], ls="--", c="orange")
    xlabel("\$t\\, /\\, J \$")
    ylabel("\$\\langle \\sigma_z(L/4) \\, \\sigma_z(3/4 L) \\rangle\$")
    title("\$correlation\\, function\$")
    layout.nice_ticks()
    savefig("figures/"*subfolder*"/corr_fct_comp.pdf")
end





show()
;
