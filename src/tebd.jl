"""
	TEBD!(mps, ham; total_time, steps, increment, observables, trotter_order=2)

Evolve the state using TEBD

See also: [`get_thermal_states`](@ref)
"""
function TEBD!(mps, ham; total_time, steps, increment, observables, trotter_order=2)
    dt = total_time/steps
    err=Array{Float64,1}(undef, steps);
    layers = prepare_layers(mps,ham,dt,trotter_order)
	data = DataFrame()
    for counter = 1:steps
		if counter % increment == 1 || increment==1
            println("step ",counter," / ",steps, "\n Dim ",maximum(length.(mps.Λ)))
			vals = Dict()
			vals["time"] = counter*dt
			vals["error"] = mps.error
			for (name, obs) in pairs(observables)
				vals[name] = [obs(mps)]
			end
			append!(data, vals)
        end
        err[counter] = apply_layers!(mps,layers)
    end
    return data, err
end


function apply_layer(sites::Vector{<:OrthogonalLinkSite}, gates, parity, truncation; isperiodic=false)
	N = length(sites)
	if isodd(N)
		error("Cell size should be even to be consistent with trotter decomposition")
	end
	itr = isodd(parity) ? (1:2:N-1) : (2:2:N-2)
	newsites = similar(sites)
	total_error = Threads.Atomic{Float64}(0.0)
	Threads.@threads for k in itr
		newsites[k], newsites[k+1], err = apply_two_site_gate(sites[k],sites[k+1], gates[k], truncation)
		Threads.atomic_add!(total_error, real(err))
	end
	if iseven(parity)
		if isperiodic
			sites[N], sites[1], error = apply_two_site_gate(sites[N], sites[1], gates[N], truncation)
			Threads.atomic_add!(total_error, real(error))
		else
			newsites[1] = copy(sites[1])
			newsites[end] = copy(sites[end])
		end
	end
	return newsites, total_error[]
end

function apply_layer!(sites::Vector{<:OrthogonalLinkSite}, gates, parity, truncation; isperiodic=false)
	N = length(sites)
	if isodd(N)
		error("Cell size should be even to be consistent with trotter decomposition")
	end
	itr = isodd(parity) ? (1:2:N-1) : (2:2:N-2)
	total_error = Threads.Atomic{Float64}(0.0)
	Threads.@threads for k in itr
		sites[k], sites[k+1], error = apply_two_site_gate(sites[k],sites[k+1], gates[k], truncation)
		Threads.atomic_add!(total_error, real(error))
	end
	if isperiodic && iseven(parity)
		sites[N], sites[1], error = apply_two_site_gate(sites[N], sites[1], gates[N], truncation)
		Threads.atomic_add!(total_error, real(error))
	end
	return sites, total_error[]
end

function apply_layer_nonunitary!(sites::Vector{<:OrthogonalLinkSite}, gates, parity, dir, truncation; isperiodic=false)
	N = length(sites)
	if isodd(N)
		error("Cell size should be even to be consistent with trotter decomposition")
	end
	itr = isodd(parity) ? (1:2:N-1) : (2:2:N-2)
	itr = (dir==-1 ? itr : reverse(itr))
	total_error = 0.0
	for k = itr
		sites[k], sites[k+1], error = apply_two_site_gate(sites[k],sites[k+1], gates[k], truncation)
		total_error += error
		if (k<N-1 && k>1) || isperiodic
			k1 = mod1.(k+dir,N)
			k2 = mod1.(k+1+dir,N)
			sites[k1], sites[k2], error = apply_two_site_gate(sites[k1],sites[k2], IdentityGate, truncation)
			total_error += error
		end
	end
	if isperiodic && iseven(parity)
		sites[N], sites[1], error = apply_two_site_gate(sites[N], sites[1], gates[N], truncation)
		total_error += error
		k1 = mod1.(N+dir,N)
		k2 = mod1.(N+1+dir,N)
		sites[k1], sites[k2], error = apply_two_site_gate(sites[k1],sites[k2], IdentityGate, truncation)
		total_error += error
	end
	return sites, total_error
end

"""
    apply_layers_nonunitary!(mps,layers)

Modify the mps by acting with the nonunitary layers of gates
"""
function apply_layers_nonunitary(sitesin::Vector{<:OrthogonalLinkSite}, layers, truncation; isperiodic=false)
    total_error = 0.0
	sites = copy(sitesin)
    for n = 1:length(layers)
        dir = isodd(n) ? 1 : -1
        _, error  = apply_layer_nonunitary!(sites, layers[n], n, dir, truncation, isperiodic=isperiodic)
        total_error += error
    end
    return sites, total_error
end

"""
    apply_layers_nonunitary!(mps,layers)

Modify the mps by acting with the nonunitary layers of gates
"""
function apply_layers(sitesin::Vector{<:OrthogonalLinkSite}, layers, truncation; isperiodic=false)
    total_error = 0.0
	sites = copy(sitesin)
    for n = 1:length(layers)
        _, error  = apply_layer!(sites, layers[n], n, truncation, isperiodic=isperiodic)
        total_error += error
    end
    return sites, total_error
end


#%% Layers
"""
	prepare_layers(gates, dt, order)

Return a list of gate layers corresponding to a single step of exp(I `gates`dt)
"""
function prepare_layers(gates,dt,order::Integer)
	if order==1
		W = st1gates(dt,gates)
	elseif order==2
		W = st2gates(dt,gates)
	elseif order==4
		W = frgates(dt,gates)
	else
		warn("Trotter order wrong or not implemented. Defaulting to order 2.")
		W = st2gates(dt,gates)
	end
	return W
end

#%% Gates
"""
	frgates(dt,gates)

Return the layers of a 4:th order Trotter scheme
"""
function frgates(dt,gates::Vector{<:AbstractSquareGate})
   theta = 1/(2-2^(1/3))
   W = Vector{Vector{AbstractSquareGate}}(undef,7)
   times = [theta/2 theta (1-theta)/2 (1-2*theta)]
   exponentiate(t) = (map(x->exp(-t*1im*dt*x),gates))
   W[1:4] = exponentiate.(times)
   # W[1] = map(x->exp(-theta*1im/2*dt*x),blocks)
   # W[2] = map(x->exp(-theta*1im*dt*x),blocks)
   # W[3] = map(x->exp(-(1-theta)*1im/2*dt*x),blocks)
   # W[4] = map(x->exp(-(1-2*theta)*1im*dt*x),blocks)
   W[5] = W[3]
   W[6] = W[2]
   W[7] = W[1]
   return W
end

"""
	st2gates(dt,gates)

Return the layers of a 2:nd order Trotter scheme
"""
function st2gates(dt,gates::Vector{<:AbstractSquareGate}) where {T,N}
   W =  Vector{Vector{AbstractSquareGate}}(undef,3)
   times = [1/2 1]
   exponentiate(t) = (map(x->exp(-t*1im*dt*x),gates))
   W[1:2] = exponentiate.(times)
   W[3] = W[1]
   return W
end

"""
	st1gates(dt,gates)

Return the layers of a 1:st order Trotter scheme
"""
function st1gates(dt,gates::Vector{<:AbstractSquareGate}) where {T,N}
   W = Vector{Vector{AbstractSquareGate}}(undef,2)
   W[1] = map(x->exp(-1im*dt*x),gates)
   W[2] = W[1]
   return W
end