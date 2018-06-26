using PyPlot


### compare exact numerical time evolution with MPS result:

E_mps = readdlm("data/quench/Gauss_L10_beta1_J1_h1/energy.txt")
E_exact = readdlm("data/quench/Gauss_L10_beta1_J1_h1/energy_exact.txt")

magnetization_mps = readdlm("data/quench/Gauss_L10_beta1_J1_h1/magnetization.txt")
magnetization_exact = readdlm("data/quench/Gauss_L10_beta1_J1_h1/magnetization_exact.txt")

corr_fct_mps = readdlm("data/quench/Gauss_L10_beta1_J1_h1/corr_fct.txt")
corr_fct_exact = readdlm("data/quench/Gauss_L10_beta1_J1_h1/corr_fct_exact.txt")


figure(1)
plot(E_mps[:,1], E_mps[:,5], label="MPS", c="k")
plot(E_exact[:,1], E_exact[:,2], ls="--", label="exact", c="orange")
xlabel("\$t\\, /\\, J \$")
ylabel("\$E(t)\$")
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
title("energy")
savefig("figures/energy_comp.pdf")

figure(2)
plot(magnetization_mps[:,1], magnetization_mps[:,5], c="k")
plot(magnetization_exact[:,1], magnetization_exact[:,2], ls="--", c="orange")
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
title("magnetization")
savefig("figures/magnetization_comp.pdf")

figure(3)
plot(corr_fct_mps[:,1], corr_fct_mps[:,5], c="k")
plot(corr_fct_exact[:,1], corr_fct_exact[:,2], ls="--", c="orange")
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_z(L/4) \\, \\sigma_z(3/4 L) \\rangle\$")
title("correlation function")
savefig("figures/corr_fct_comp.pdf")





show()
;
