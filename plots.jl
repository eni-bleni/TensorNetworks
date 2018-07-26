using layout
using PyPlot



###-----------------------------------------------------------------------------
### convergence with increasing D:

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
xlim(0,7)
xlabel("\$t\\, /\\, J \$")
ylabel("\$E(t)\$")
title("\$energy\$")
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
layout.nice_ticks()
savefig("figures/"*subfolder*"/energy.pdf")

figure(2)
xlim(0,7)
ylim(-0.425,-0.4)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
title("\$magnetization\$")
ax = subplot(111)
ax[:set_yticks]([-0.42, -0.41, -0.4])
layout.nice_ticks()
savefig("figures/"*subfolder*"/magnetization.pdf")

figure(3)
xlim(0,7)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_z(L/4) \\, \\sigma_z(3/4 L) \\rangle\$")
title("\$correlation\\, function\$")
layout.nice_ticks()
ax = subplot(111)
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-6,4), useOffset=true)
savefig("figures/"*subfolder*"/corr_fct.pdf")



###-----------------------------------------------------------------------------
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



###-----------------------------------------------------------------------------
### Temperature dependence:

subfolder = "1e-1shortdetailGauss_L50_D200_J1_h1"

E_mps, header = readdlm("data/quench/"*subfolder*"/energy.txt", header=true)
magnetization_mps, header = readdlm("data/quench/"*subfolder*"/magnetization.txt", header=true)
corr_fct_mps, header = readdlm("data/quench/"*subfolder*"/corr_fct.txt", header=true)
betas = include_string(join(header[3:end]))

for i = 1:length(betas)
    beta = betas[i]

    figure(7)
    plot(E_mps[:,1], E_mps[:,i+1]/49, label="\$\\beta_{th}\\, / \\,J = $beta\$")
    figure(8)
    plot(magnetization_mps[:,1], magnetization_mps[:,i+1])
    figure(9)
    plot(corr_fct_mps[:,1], corr_fct_mps[:,i+1])
end

figure(7)
xlabel("\$t\\, /\\, J \$")
ylabel("\$E(t)\\, / \\,L\$")
title("\$energy\$")
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
layout.nice_ticks()
savefig("figures/"*subfolder*"/energy.pdf")

figure(8)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
title("\$magnetization\$")
layout.nice_ticks()
savefig("figures/"*subfolder*"/magnetization.pdf")

figure(9)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_z(L/4) \\, \\sigma_z(3/4 L) \\rangle\$")
title("\$correlation\\, function\$")
layout.nice_ticks()
ax = subplot(111)
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-6,4), useOffset=true)
savefig("figures/"*subfolder*"/corr_fct.pdf")



###-----------------------------------------------------------------------------
### system size (L) dependence:

subfolder = "1e-1shortGauss_beta0.01_Lstudies"
sizes = [80,120,200]
lstyles = ["-", "--", "-."]

for i = 1:length(sizes)
    L = sizes[i]
    E_mps, header = readdlm("data/quench/"*subfolder*"/L"*string(L)*"/energy.txt", header=true)
    magnetization_mps, header = readdlm("data/quench/"*subfolder*"/L"*string(L)*"/magnetization.txt", header=true)
    corr_fct_mps, header = readdlm("data/quench/"*subfolder*"/L"*string(L)*"/corr_fct.txt", header=true)

    figure(10)
    plot(E_mps[:,1], E_mps[:,2]/(L-1), label="\$L = $L\$", ls=lstyles[i])
    figure(11)
    plot(magnetization_mps[:,1], magnetization_mps[:,2], ls=lstyles[i])
    figure(12)
    plot(corr_fct_mps[:,1], corr_fct_mps[:,2], ls=lstyles[i])
end

figure(10)
xlabel("\$t\\, /\\, J \$")
ylabel("\$E(t)\\, / \\,L\$")
title("\$energy\$")
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
layout.nice_ticks()
ax = subplot(111)
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
savefig("figures/"*subfolder*"/energy.pdf")

figure(11)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
title("\$magnetization\$")
layout.nice_ticks()
ax = subplot(111)
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
savefig("figures/"*subfolder*"/magnetization.pdf")

figure(12)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_z(L/4) \\, \\sigma_z(3/4 L) \\rangle\$")
title("\$correlation\\, function\$")
layout.nice_ticks()
savefig("figures/"*subfolder*"/corr_fct.pdf")



show()
;
