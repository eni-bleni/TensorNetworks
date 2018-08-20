using layout # run this module at first as:  include("layout.jl")
using PyPlot

function energy_layout(set_xlim)
    if set_xlim xlim(0,4) end
    xlabel("\$t\\, /\\, J \$")
    ylabel("\$E(t)\\, / \\,L\$")
    title("\$energy\$")
    legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
    layout.nice_ticks()
end

function magnetization_layout(set_xlim)
    if set_xlim xlim(0,4) end
    xlabel("\$t\\, /\\, J \$")
    ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
    title("\$magnetization\$")
    layout.nice_ticks()
end

function corr_fct_layout(set_xlim)
    if set_xlim xlim(0,4) end
    xlabel("\$t\\, /\\, J \$")
    ylabel("\$\\langle \\sigma_z(L/4) \\, \\sigma_z(3/4 L) \\rangle\$")
    title("\$correlation\\, function\$")
    layout.nice_ticks()
end

function error_layout(set_xlim)
    if set_xlim xlim(0,5) end
    xlabel("\$t\\, /\\, J \$")
    title("\$error\$")
    layout.nice_ticks()
end

function read_and_plot_Tdependence(fig_num, plot_scaled_mag=false)
    f = open("data/quench/"*subfolder*"/opvalues.txt")
    lines = readlines(f)
    close(f)
    sep_inds = findin(lines, [""])

    thermal_vals = readdlm("data/quench/"*subfolder*"/thermalization.txt")

    for i = 1:length(sep_inds)
        counter = 1

        if i==1
            steps = parse(split(lines[1])[8])
            E_mps = Array{Float64}(steps, 2)
            magnetization_mps = Array{Float64}(steps, 2)
            corr_fct_mps = Array{Float64}(steps, 2)
            error_mps = Array{Float64}(steps, 2)
            beta = include_string(split(lines[1])[4])
            L = include_string(split(lines[1])[2])
            for l = 3 : sep_inds[1]-1
                line = parse.(split(lines[l]))
                E_mps[counter,:] = [line[1] line[2]]
                magnetization_mps[counter,:] = [line[1] line[3]]
                corr_fct_mps[counter,:] = [line[1] line[4]]
                error_mps[counter,:] = [line[1] line[5]]
                counter += 1
            end
        else
            steps = parse(split(lines[sep_inds[i-1]+1])[8])
            E_mps = Array{Float64}(steps, 2)
            magnetization_mps = Array{Float64}(steps, 2)
            corr_fct_mps = Array{Float64}(steps, 2)
            error_mps = Array{Float64}(steps, 2)
            beta = include_string(split(lines[sep_inds[i-1]+1])[4])
            L = include_string(split(lines[sep_inds[i-1]+1])[2])
            for l = sep_inds[i-1]+3 : sep_inds[i]-1
                line = parse.(split(lines[l]))
                E_mps[counter,:] = [line[1] line[2]]
                magnetization_mps[counter,:] = [line[1] line[3]]
                corr_fct_mps[counter,:] = [line[1] line[4]]
                error_mps[counter,:] = [line[1] line[5]]
                counter += 1
            end
        end

        figure(fig_num)
        plot(E_mps[:,1], E_mps[:,2]/(L-1), label="\$\\beta_{th}\\, / \\,J = $beta\$")
        figure(fig_num+1)
        plot(magnetization_mps[:,1], magnetization_mps[:,2])
        axhline(thermal_vals[i,5], ls="--",c="b")
        figure(fig_num+2)
        plot(corr_fct_mps[:,1], corr_fct_mps[:,2])
        figure(fig_num+3)
        plot(error_mps[:,1], error_mps[:,2])
        if plot_scaled_mag
            figure(fig_num+4)
            plot(magnetization_mps[:,1]/beta, magnetization_mps[:,2])
        end
    end
end


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
### Temperature dependence at criticality:

subfolder = "atCriticality"
read_and_plot_Tdependence(7,true)


figure(7)
energy_layout(true)
savefig("figures/"*subfolder*"/energy.pdf")

figure(8)
# yscale("symlog", linthreshy=0.1)
magnetization_layout(true)
savefig("figures/"*subfolder*"/magnetization.pdf")

figure(9)
corr_fct_layout(true)
ax = subplot(111)
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-6,4), useOffset=true)
savefig("figures/"*subfolder*"/corr_fct.pdf")

figure(10)
error_layout(true)
savefig("figures/"*subfolder*"/error.pdf")

figure(11)
xlabel("\$t\\, /\\, \\beta \$")
ylabel("\$\\langle \\sigma_x(L/2) \\rangle\$")
title("\$magnetization\$")
layout.nice_ticks()
savefig("figures/"*subfolder*"/magnetization_tT.pdf")



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

    figure(12)
    plot(E_mps[:,1], E_mps[:,2]/(L-1), label="\$L = $L\$", ls=lstyles[i])
    figure(13)
    plot(magnetization_mps[:,1], magnetization_mps[:,2], ls=lstyles[i])
    figure(14)
    plot(corr_fct_mps[:,1], corr_fct_mps[:,2], ls=lstyles[i])
end

figure(12)
energy_layout(false)
ax = subplot(111)
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
savefig("figures/"*subfolder*"/energy.pdf")

figure(13)
magnetization_layout(false)
ax = subplot(111)
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-2,4), useOffset=true)
savefig("figures/"*subfolder*"/magnetization.pdf")

figure(14)
corr_fct_layout(false)
savefig("figures/"*subfolder*"/corr_fct.pdf")



###-----------------------------------------------------------------------------
### correlation fct in dependence on beta:

f = open("data/correlations/corr_fcts_offcrit.txt")
lines = readlines(f)
close(f)

sep_inds = findin(lines, [""])

for i = 1:length(sep_inds)
    corr = Array{Float64}(parse(split(lines[1])[2])-1, 2)

    if i==1
        beta = include_string(split(lines[1])[4])
        for l = 3 : sep_inds[1]-1
            line = parse.(split(lines[l]))
            corr[line[1],:] = [line[1] line[2]]
        end
    else
        beta = include_string(split(lines[sep_inds[i-1]+1])[4])
        for l = sep_inds[i-1]+3 : sep_inds[i]-1
            line = parse.(split(lines[l]))
            corr[line[1],:] = [line[1] line[2]]
        end
    end

    figure(15)
    semilogy(corr[:,1], corr[:,2], label="\$\\beta_{th}\\, / \\,J = $beta\$")
end

figure(15)
xlabel("\$ m \$")
ylabel("\$\\vert \\langle \\sigma_z(1) \\, \\sigma_z(1+m) \\rangle \\vert\$")
title("\$correlation\\, function\$")
# legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)
layout.nice_ticks()
savefig("figures/correlations/corr_fct_distance_offcrit.pdf")



###-----------------------------------------------------------------------------
### magnitude dependence:
E0 = -0.820980161008688
mag0 = -0.4058378480098809

# new format:
subfolder = "magnitude_studies"
f = open("data/quench/"*subfolder*"/opvalues.txt")
lines = readlines(f)
close(f)
sep_inds = findin(lines, [""])

for i = 1:length(sep_inds)
    counter = 1

    if i==1
        steps = parse(split(lines[1])[8])
        E_mps = Array{Float64}(steps, 2)
        magnetization_mps = Array{Float64}(steps, 2)
        corr_fct_mps = Array{Float64}(steps, 2)
        error_mps = Array{Float64}(steps, 2)
        delta = include_string(split(lines[1])[14])
        L = include_string(split(lines[1])[2])
        for l = 3 : sep_inds[1]-1
            line = parse.(split(lines[l]))
            E_mps[counter,:] = [line[1] line[2]]
            magnetization_mps[counter,:] = [line[1] line[3]]
            corr_fct_mps[counter,:] = [line[1] line[4]]
            error_mps[counter,:] = [line[1] line[5]]
            counter += 1
        end
    else
        steps = parse(split(lines[sep_inds[i-1]+1])[8])
        E_mps = Array{Float64}(steps, 2)
        magnetization_mps = Array{Float64}(steps, 2)
        corr_fct_mps = Array{Float64}(steps, 2)
        error_mps = Array{Float64}(steps, 2)
        delta = include_string(split(lines[sep_inds[i-1]+1])[14])
        L = include_string(split(lines[sep_inds[i-1]+1])[2])
        for l = sep_inds[i-1]+3 : sep_inds[i]-1
            line = parse.(split(lines[l]))
            E_mps[counter,:] = [line[1] line[2]]
            magnetization_mps[counter,:] = [line[1] line[3]]
            corr_fct_mps[counter,:] = [line[1] line[4]]
            error_mps[counter,:] = [line[1] line[5]]
            counter += 1
        end
    end

    figure(16)
    plot(E_mps[:,1], E_mps[:,2]/(L-1), label="\$\\delta = $delta\$")
    figure(17)
    plot(E_mps[:,1], (E_mps[:,2]/(L-1)-E0) / (delta^2))
    figure(18)
    plot(E_mps[:,1], (E_mps[:,2]/(L-1)-E0) / delta)
    figure(19)
    plot(magnetization_mps[:,1], magnetization_mps[:,2])
    figure(20)
    plot(magnetization_mps[:,1], (magnetization_mps[:,2]-mag0)/delta)
    figure(21)
    plot(corr_fct_mps[:,1], corr_fct_mps[:,2])
    figure(22)
    plot(error_mps[:,1], error_mps[:,2])
end


figure(16)
axis([0,4,-0.9,-0.7])
ax = subplot(111)
ax[:set_yticks]([-0.7,-0.75,-0.8,-0.85,-0.9])
xlabel("\$t\\, /\\, J \$")
ylabel("\$E(t)\\, / \\,L\$")
title("\$energy\$")
legend(loc = "lower right", ncol=2, numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1, title="\$\\beta_{th} = 0.5\$")
layout.nice_ticks()
savefig("figures/"*subfolder*"/energy.pdf")

figure(17)
axis([0,4,-4.5,0.4])
xlabel("\$t\\, /\\, J \$")
ylabel("\$[E(t)-E(0)]\\, / \\,L\\, / \\,\\delta^2\$")
title("\$energy\$")
# legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1, title="\$\\beta = 0.5\$")
layout.nice_ticks()
savefig("figures/"*subfolder*"/energy_norm2.pdf")

figure(18)
axis([0,4,-0.3,0.1])
xlabel("\$t\\, /\\, J \$")
ylabel("\$[E(t)-E(0)]\\, / \\,L\\, / \\,\\delta\$")
title("\$energy\$")
# legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1, title="\$\\beta = 0.5\$")
layout.nice_ticks()
savefig("figures/"*subfolder*"/energy_norm.pdf")

figure(19)
axis([0,4,-0.5,-0.29])
ax = subplot(111)
ax[:set_yticks]([-0.5,-0.45,-0.4,-0.35,-0.3])
magnetization_layout(false)
savefig("figures/"*subfolder*"/magnetization.pdf")

figure(20)
axis([0,4,-0.2,0.06])
# ax = subplot(111)
# ax[:set_yticks]([-0.5,-0.45,-0.4,-0.35,-0.3])
xlabel("\$t\\, /\\, J \$")
ylabel("\$[\\langle \\sigma_x(L/2) \\rangle(t) - \\langle \\sigma_x(L/2) \\rangle(t=0)] / \\delta \$")
title("\$magnetization\$")
layout.nice_ticks()
savefig("figures/"*subfolder*"/magnetization_norm.pdf")

figure(21)
axis([0,4,-0.8e-9,0.1e-10])
corr_fct_layout(false)
ax = subplot(111)
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-6,4), useOffset=true)
savefig("figures/"*subfolder*"/corr_fct.pdf")

figure(22)
error_layout(true)
savefig("figures/"*subfolder*"/error.pdf")



###-----------------------------------------------------------------------------
### Temperature dependence off criticality:

subfolder = "offCriticality"
read_and_plot_Tdependence(23)


figure(23)
energy_layout(true)
savefig("figures/"*subfolder*"/energy.pdf")

figure(24)
magnetization_layout(true)
savefig("figures/"*subfolder*"/magnetization.pdf")

figure(25)
corr_fct_layout(true)
ax = subplot(111)
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-6,4), useOffset=true)
savefig("figures/"*subfolder*"/corr_fct.pdf")

figure(26)
error_layout(true)
savefig("figures/"*subfolder*"/error.pdf")



###-----------------------------------------------------------------------------
### Temperature dependence instantaneous quench:

subfolder = "Instantaneous_Tstudies"
read_and_plot_Tdependence(27)


figure(27)
energy_layout(true)
savefig("figures/"*subfolder*"/energy.pdf")

figure(28)
magnetization_layout(true)
savefig("figures/"*subfolder*"/magnetization.pdf")

figure(29)
corr_fct_layout(true)
ax = subplot(111)
ax[:ticklabel_format](axis="y", style="scientific", scilimits=(-6,4), useOffset=true)
savefig("figures/"*subfolder*"/corr_fct.pdf")

figure(30)
error_layout(true)
savefig("figures/"*subfolder*"/error.pdf")





# #######
show()
;
