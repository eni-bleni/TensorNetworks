using layout
using PyPlot
# using Interpolations
using LsqFit
# using Optim


### read out data:

subfolder = "1e-1shortdetailGauss_L50_beta0.5_J1_h1"

magnetization_mps, header = readdlm("data/quench/"*subfolder*"/magnetization.txt", header=true)
bondDims = include_string(join(header[3:end]))

time = Float64.(magnetization_mps[:,1])
sx_t = Float64.(magnetization_mps[:,length(bondDims)+1])

t_min = 1.5
t_max = 4.5
ind_min = maximum(find(time .<= t_min))
ind_max = minimum(find(time .>= t_max))

model(t, p) = p[1].*exp.(-p[2].*t).*sin.(p[3].*(t-p[4]))+p[5]
p0 = [0.5, 0.1, 10.0, 2.0, -0.5]
fit = curve_fit(model, time[ind_min:ind_max], sx_t[ind_min:ind_max], p0)
# itp = interpolate(sx_t)

figure(1)
plot(time, sx_t)
plot(time, model(time, fit.param), ls="--", label="fit")
axis([t_min, t_max, -0.407, -0.4025])
legend(loc = "best", numpoints=3, frameon = 0, fancybox = 0, columnspacing = 1)

println("Re(w) = ", fit.param[3], ",  Im(w) = ", fit.param[2])


show()
;
