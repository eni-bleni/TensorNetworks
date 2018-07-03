using PyPlot
using PyCall
@pyimport matplotlib.transforms as mpltrafo



### LAYOUT choices:

function nice_ticks()
    ax = subplot(111)
    ax[:get_xaxis]()[:set_tick_params](direction="in", bottom=1, top=1)
    ax[:get_yaxis]()[:set_tick_params](direction="in", left=1, right=1)

    for l in ax[:get_xticklines]()
        l[:set_markersize](8)
        l[:set_markeredgewidth](2.0)
    end
    for l in ax[:get_yticklines]()
        l[:set_markersize](8)
        l[:set_markeredgewidth](2.0)
    end
    for l in ax[:yaxis][:get_minorticklines]()
        l[:set_markersize](4)
        l[:set_markeredgewidth](1.5)
    end
    for l in ax[:xaxis][:get_minorticklines]()
        l[:set_markersize](4)
        l[:set_markeredgewidth](1.5)
    end

    ax[:set_position](mpltrafo.Bbox([[0.16, 0.12], [0.95, 0.94]]))
end

linew = 2
rc("font", size = 18) #fontsize of axis labels (numbers)
rc("axes", labelsize = 20, lw = linew) #fontsize of axis labels (symbols)
rc("lines", mew = 2, lw = linew, markeredgewidth = 2)
rc("patch", ec = "k")
rc("xtick.major", pad = 7)
rc("ytick.major", pad = 7)

PyCall.PyDict(matplotlib["rcParams"])["mathtext.fontset"] = "cm"
PyCall.PyDict(matplotlib["rcParams"])["mathtext.rm"] = "serif"
PyCall.PyDict(matplotlib["rcParams"])["figure.figsize"] = [8.0, 6.0]



### read out data:

subfolder = ""

E_mps, header = readdlm("data/quench/"*subfolder*"/energy.txt", header=true)
E_exact = readdlm("data/quench/"*subfolder*"/energy_exact.txt")

magnetization_mps, header = readdlm("data/quench/"*subfolder*"/magnetization.txt", header=true)
magnetization_exact = readdlm("data/quench/"*subfolder*"/magnetization_exact.txt")

corr_fct_mps, header = readdlm("data/quench/"*subfolder*"/corr_fct.txt", header=true)
corr_fct_exact = readdlm("data/quench/"*subfolder*"/corr_fct_exact.txt")

bondDims = include_string(join(header[3:end]))



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
nice_ticks()
savefig("figures/"*subfolder*"/energy.pdf")

figure(2)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
title("\$magnetization\$")
nice_ticks()
savefig("figures/"*subfolder*"/magnetization.pdf")

figure(3)
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_z(L/4) \\, \\sigma_z(3/4 L) \\rangle\$")
title("\$correlation\\, function\$")
nice_ticks()
savefig("figures/"*subfolder*"/corr_fct.pdf")



### compare exact numerical time evolution with MPS result (for highest D):

figure(4)
plot(E_mps[:,1], E_mps[:,length(bondDims)+1], label="MPS", c="k")
plot(E_exact[:,1], E_exact[:,2], ls="--", label="exact", c="orange")
xlabel("\$t\\, /\\, J \$")
ylabel("\$E(t)\$")
title("\$energy\$")
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
nice_ticks()
savefig("figures/"*subfolder*"/energy_comp.pdf")

figure(5)
plot(magnetization_mps[:,1], magnetization_mps[:,length(bondDims)+1], c="k")
plot(magnetization_exact[:,1], magnetization_exact[:,2], ls="--", c="orange")
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
title("\$magnetization\$")
nice_ticks()
savefig("figures/"*subfolder*"/magnetization_comp.pdf")

figure(6)
plot(corr_fct_mps[:,1], corr_fct_mps[:,length(bondDims)+1], c="k")
plot(corr_fct_exact[:,1], corr_fct_exact[:,2], ls="--", c="orange")
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_z(L/4) \\, \\sigma_z(3/4 L) \\rangle\$")
title("\$correlation\\, function\$")
nice_ticks()
savefig("figures/"*subfolder*"/corr_fct_comp.pdf")






show()
;
