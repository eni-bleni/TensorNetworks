using layout
using PyPlot
# using Interpolations
using LsqFit
# using Optim
println("\n---fourier.jl------------------------------------")


### read out data:

subfolder = "1e-1shortdetailGauss_L50_beta0.5_J1_h1"
beta = include_string(split(split(subfolder,"_")[3],"beta")[2]) # = 0.5

magnetization_mps, header = readdlm("data/quench/"*subfolder*"/magnetization.txt", header=true)
bondDims = include_string(join(header[3:end]))

time = Float64.(magnetization_mps[:,1])
sx_t = Float64.(magnetization_mps[:,length(bondDims)+1])

t_min = 1.5
t_max = 4.0
ind_min = maximum(find(time .<= t_min))
ind_max = minimum(find(time .>= t_max))

## fit model fcts for damped oscillations:
model_exp(t, p) = p[1].*exp.(-p[2].*t) .* sin.(p[3].*(t-p[4])) .* t.^p[5] + p[6]
p0_exp = [0.5, 0.1, 10.0, 2.0, 0.0, -0.5]
model_power(t, p) = p[1] .* sin.(p[2].*(t-p[3])) .* t.^p[4] + p[5]
p0_power = [0.5, 10.0, 2.0, 0.0, -0.5]

## choose fit type:
decay_exp = false
if decay_exp # exponential + power law decay
    model = model_exp
    p0 = p0_exp
else         # only power law decay
    model = model_power
    p0 = p0_power
end

fit = curve_fit(model, time[ind_min:ind_max], sx_t[ind_min:ind_max], p0)

figure(1)
plot(time, sx_t/beta, label="\$\\beta_{th}\\, / \\,J = $beta\$")
plot(time, model(time, fit.param)/beta, ls="--", c="k")

if decay_exp
    figure(2)
    plot(fit.param[3], fit.param[2], ls="", marker="s")
end

println("\nbeta_th = ", beta)
if decay_exp
    println("Re(w) = ", fit.param[3], ",  Im(w) = ", fit.param[2])
    println("beta = ", fit.param[5])
else
    println("Re(w) = ", fit.param[2], ",  Im(w) = ", 0)
    println("beta = ", fit.param[4])
end



### Temperature dependence: ----------------------------------------------------
subfolder = "1e-1shortdetailGauss_L50_D200_J1_h1"

magnetization_mps, header = readdlm("data/quench/"*subfolder*"/magnetization.txt", header=true)
betas = include_string(join(header[3:end]))

for i = 1:length(betas)
    beta = betas[i]

    time = Float64.(magnetization_mps[:,1])
    sx_t = Float64.(magnetization_mps[:,i+1])

    ind_min = maximum(find(time .<= t_min))
    ind_max = minimum(find(time .>= t_max))

    fit = curve_fit(model, time[ind_min:ind_max], sx_t[ind_min:ind_max], p0)

    figure(1)
    plot(time, sx_t/beta, label="\$\\beta_{th}\\, / \\,J = $beta\$")
    plot(time, model(time, fit.param)/beta, ls="--", c="k")

    if decay_exp
        figure(2)
        plot(fit.param[3], fit.param[2], ls="", marker="s")
    end

    println("\nbeta_th = ", beta)
    if decay_exp
        println("Re(w) = ", fit.param[3], ",  Im(w) = ", fit.param[2])
        println("beta = ", fit.param[5])
    else
        println("Re(w) = ", fit.param[2], ",  Im(w) = ", 0)
        println("beta = ", fit.param[4])
    end
end


figure(1)
axis([t_min, t_max, -1.01, -0.8])
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\\, /\\, \\beta_{th}\$")
title("\$magnetization\$")
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
ax = subplot(111)
ax[:set_yticks]([-1, -0.95, -0.9, -0.85, -0.8])
layout.nice_ticks()
savefig("figures/"*subfolder*"/magnetization_fourier.pdf")

if decay_exp
    figure(2)
    xlabel("\$Re(\\omega)\$")
    ylabel("\$Im(\\omega)\$")
    layout.nice_ticks()
end

show()
;
