using layout
using PyPlot
# using Interpolations
using LsqFit
# using Optim
println("\n---fourier.jl------------------------------------")


### read out data:
subfolder = "atCriticality"

t_min = 1.5
t_max = 4.0

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


### Temperature dependence: ----------------------------------------------------
f = open("data/quench/thermal/"*subfolder*"/opvalues.txt")
lines = readlines(f)
close(f)
sep_inds = findin(lines, [""])

for i = 1:length(sep_inds)
    counter = 1

    if i==1
        num_steps = parse(split(lines[1])[8])
        magnetization_mps = Array{Float64}(num_steps, 2)
        L = include_string(split(lines[1])[2])
        beta_th = include_string(split(lines[1])[4])
        for l = 3 : sep_inds[1]-1
            line = parse.(split(lines[l]))
            magnetization_mps[counter,:] = [line[1] line[3]]
            counter += 1
        end
    else
        num_steps = parse(split(lines[sep_inds[i-1]+1])[8])
        magnetization_mps = Array{Float64}(num_steps, 2)
        L = include_string(split(lines[sep_inds[i-1]+1])[2])
        beta_th = include_string(split(lines[sep_inds[i-1]+1])[4])
        for l = sep_inds[i-1]+3 : sep_inds[i]-1
            line = parse.(split(lines[l]))
            magnetization_mps[counter,:] = [line[1] line[3]]
            counter += 1
        end
    end

    time = Float64.(magnetization_mps[:,1])
    sx_t = Float64.(magnetization_mps[:,2])

    ind_min = maximum(find(time .<= t_min))
    ind_max = minimum(find(time .>= t_max))

    fit = curve_fit(model, time[ind_min:ind_max], sx_t[ind_min:ind_max], p0)

    figure(1)
    plot(time, sx_t, label="\$\\beta_{th}\\, / \\,J = $beta_th\$")
    plot(time, model(time, fit.param), ls="--", c="k")

    if decay_exp
        figure(2)
        plot(fit.param[3], fit.param[2], ls="", marker="s")
    end

    println("\nbeta_th = ", beta_th)
    if decay_exp
        println("Re(w) = ", fit.param[3], ",  Im(w) = ", fit.param[2])
        println("alpha = ", fit.param[5])
    else
        println("Re(w) = ", fit.param[2], ",  Im(w) = ", 0)
        println("alpha = ", fit.param[4])
    end
end


## PLOTS:
figure(1)
# xlim(t_min, t_max)
axis([t_min, t_max, -0.7, -0.0])
xlabel("\$t\\, /\\, J \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
title("\$magnetization\$")
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
# legend(bbox_to_anchor=(0.5, 0.5), numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
# ax = subplot(111)
# ax[:set_yticks]([-1, -0.95, -0.9, -0.85, -0.8])
layout.nice_ticks()
savefig("figures/thermal/"*subfolder*"/magnetization_fourier.pdf")

if decay_exp
    figure(2)
    xlabel("\$Re(\\omega)\$")
    ylabel("\$Im(\\omega)\$")
    layout.nice_ticks()
end

show()
;
