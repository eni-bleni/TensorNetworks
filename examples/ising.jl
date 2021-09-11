using TensorNetworks
# %%
N = 20
d = 2 
Dmax = 20
prec = 1e-14
truncation = TruncationArgs(Dmax, prec, false)

J = 1.0
h = .5
g = 0.001
hammpo = IsingMPO(N,J,h,g)
hamgates = isingHamGates(N,J,h,g)
# %% DMRG
n_states= 2
initialmps = randomLCROpenMPS(N,d, Dmax; truncation = truncation)
@time states, energies = eigenstates(hammpo, initialmps, n_states, precision = prec, alpha=5);

# %% Expectation values

magGate = Gate(sz)
mag = [real.(expectation_values(state, magGate)) for state in states];

domainwallGate = Gate(reshape(kron(sz,sz), (2,2,2,2)))
domainwall = [real.(expectation_values(state, domainwallGate)) for state in states];

hamgate = hamgates[2]
es = [real.(expectation_values(state, hamgate)) for state in states];

# %%
using Plots
plot(mag,show=true)
plot(domainwall,show=true)
# %% Thermal states
