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
		if counter % increment == 1
            println("step ",counter," / ",steps, "\n Dim ",maximum(length.(mps.Λ)))
			vals = Dict()
			vals["time"] = counter*dt
			vals["error"] = mps.error[]
			for (name, obs) in pairs(observables)
				vals[name] = [obs(mps)]
			end
			append!(data, vals)
        end
        err[counter] = apply_layers!(mps,layers)
    end
    return data, err
end


"""
	apply_layer!(Γout, Λout, Γin, Λin, gates, parity, truncation)

Modify the list of tensor by applying the gates

See also: [`apply_layer`](@ref), [`apply_layer_distributed`](@ref)
"""
function apply_layer!(Γout, Λout, Γin, Λin, gates, parity, truncation)
	N = length(Γin)
	if isodd(N)
		error("Cell size should be even to be consistent with trotter decomposition")
	end
	itr = isodd(parity) ? (1:2:N-1) : (2:2:N-2)
	total_error = 0.0
	for k = itr
		Γout[k], Λout[k+1], Γout[k+1], error = apply_two_site_gate(view(Γin, k:k+1), view(Λin, k:k+2), gates[k], truncation)
		total_error += error
	end
	return total_error
end

"""
	apply_layer_distributed(Γout, Λout, Γin, Λin, gates, parity, truncation)

Return the list of tensor acted on by the gates

See also: [`apply_layer!`](@ref)
"""
function apply_layer_distributed(Γin, Λin, gates, parity, truncation)
	N = length(Γin)
	if isodd(N)
		error("Cell size should be even to be consistent with trotter decomposition")
	end
	itr = isodd(parity) ? (1:2:N-1) : (2:2:N-2)
	total_error = 0.0
	function apply_gate(k)
		apply_two_site_gate(Γin[k:k+1], Λin[k:k+2], gates[k], truncation)
	end
	ΓlΓe = pmap(k->apply_gate(k),itr)
	Γout = similar(Γin)
	Λout = deepcopy(Λin)
	total_error = 0.0
	if iseven(parity)
		Γout[1] = Γin[1]
		Γout[end] = Γin[end]
	end
	for k in 1:length(itr)
		Γout[itr[k]] = ΓlΓe[k][1]
		Λout[itr[k]+1] = ΓlΓe[k][2]
		Γout[itr[k]+1] = ΓlΓe[k][3]
		total_error += ΓlΓe[k][4]
	end
	return Γout, Λout, total_error
end

"""
	apply_layer(Γout, Λout, Γin, Λin, gates, parity, truncation)

Return the list of tensors acted on by the gates. Threaded

See also: [`apply_layer!`](@ref), [`apply_layer_distributed`](@ref)
"""
function apply_layer(Γin, Λin, gates, parity, truncation)
	N = length(Γin)
	if isodd(N)
		error("Cell size should be even to be consistent with trotter decomposition")
	end
	itr = isodd(parity) ? (1:2:N-1) : (2:2:N-2)
	total_error = 0.0
	Γout = similar(Γin)
	Λout = deepcopy(Λin)
	total_error = Threads.Atomic{Float64}(0.0)
	Threads.@threads for k in itr
		Γout[k], Λout[k+1], Γout[k+1], err = apply_two_site_gate(Γin[k:k+1], Λin[k:k+2], gates[k], truncation)
		Threads.atomic_add!(total_error,err::Float64)
	end
	if iseven(parity)
		Γout[1] = Γin[1]
		Γout[end] = Γin[end]
	end
	return Γout, Λout, total_error[]::Float64
end

"""
	apply_identity_layer(Γin, Λin, parity, truncation)

Apply the identity layer
"""
function apply_identity_layer(Γin, Λin, parity, truncation)
	N = length(Γin)
	if isodd(N)
		error("Cell size should be even to be consistent with trotter decomposition")
	end
	itr = isodd(parity) ? (1:2:N-1) : (2:2:N-2)
	Γout = similar(Γin)
	Λout = deepcopy(Λin)
	total_error = Threads.Atomic{Float64}(0.0)
	Threads.@threads for k in itr
		Γout[k], Λout[k+1], Γout[k+1], err = apply_two_site_identity(Γin[k:k+1], Λin[k:k+2], truncation)
		Threads.atomic_add!(total_error,real(err))
	end
	if iseven(parity)
		Γout[1] = Γin[1]
		Γout[end] = Γin[end]
	end
	return Γout, Λout, total_error[]
end

"""
	apply_identity_layer_distributed(Γin, Λin, parity, truncation)

Apply the identity layer distributed over all workers
"""
function apply_identity_layer_distributed(Γin, Λin, parity, truncation)
	N = length(Γin)
	if isodd(N)
		error("Cell size should be even to be consistent with trotter decomposition")
	end
	itr = isodd(parity) ? (1:2:N-1) : (2:2:N-2)
	total_error = 0.0
	function apply_gate(k)
		apply_two_site_identity(Γin[k:k+1], Λin[k:k+2], truncation)
	end
	ΓlΓe = pmap(k->apply_gate(k),itr)
	Γout = similar(Γin)
	Λout = deepcopy(Λin)
	total_error = 0.0
	if iseven(parity)
		Γout[1] = Γin[1]
		Γout[end] = Γin[end]
	end
	for k in 1:length(itr)
		Γout[itr[k]] = ΓlΓe[k][1]
		Λout[itr[k]+1] = ΓlΓe[k][2]
		Γout[itr[k]+1] = ΓlΓe[k][3]
		total_error += ΓlΓe[k][4]
	end
	return Γout, Λout, total_error
end

"""
	apply_layer_nonunitary!(Γin, Λin, parity, truncation)

Apply the nonunitary layer
"""
function apply_layer_nonunitary!(Γ, Λ, gates, parity, dir, truncation)
	N = length(Γ)
	if isodd(N)
		error("Cell size should be even to be consistent with trotter decomposition")
	end
	itr = isodd(parity) ? (1:2:N-1) : (2:2:N-2)
	itr = dir==-1 ? itr : Base.reverse(itr)
	total_error = 0.0
	for k = itr
		Γ[k], Λ[k+1], Γ[k+1], error = apply_two_site_gate(Γ[k:k+1], Λ[k:k+2], gates[k], truncation)
		total_error += error
		if k<N-1 && k>1
			Γ[k+dir], Λ[k+1+dir], Γ[k+1+dir], error = apply_two_site_identity(Γ[(k:k+1) .+ dir], Λ[(k:k+2) .+ dir], truncation)
		end
		total_error+=error
	end
	return total_error
end

#%% Layers
"""
	prepare_layers!(gates, dt, order)

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
function frgates(dt,gates)
   theta = 1/(2-2^(1/3))
   W = Array{Array{Array{Complex{Float64},4},1},1}(undef,7)
   d =size(gates[1],1)
   blocks = gate_to_block.(gates)
   times = [theta/2 theta (1-theta)/2 (1-2*theta)]
   exponentiate(t) = block_to_gate.(map(x->exp(-t*1im*dt*x),blocks))
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
function st2gates(dt,gates)
   W = Array{Array{Array{Complex{Float64},4},1},1}(undef,3)
   d =size(gates[1],1)
   blocks = gate_to_block.(gates)
   times = [1/2 1]
   exponentiate(t) = block_to_gate.(map(x->exp(-t*1im*dt*x),blocks))
   W[1:2] = exponentiate.(times)
   W[3] = W[1]
   return W
end

"""
	st1gates(dt,gates)

Return the layers of a 1:st order Trotter scheme
"""
function st1gates(dt,gates)
   W = Array{Array{Array{Complex{Float64},4},1},1}(undef,2)
   d = size(gates[1],1)
   blocks = gate_to_block.(gates)
   W[1] = map(x->reshape(exp(-1im*dt*x),d,d,d,d),blocks)
   W[2] = W[1]
   return W
end
