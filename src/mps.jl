
Base.length(mps::AbstractMPS) = length(mps.Γ)
Base.getindex(mps::AbstractMPS, I) = [mps[i] for i in I]

function prepare_layers(mps::AbstractMPS, gs::Vector{<:AbstractSquareGate}, dt, trotter_order)
	gates = (mps.purification ? auxillerate.(gs) : gs)
	return prepare_layers(gates,dt,trotter_order)
end


"""
	get_thermal_states(mps, hamGates, betas, dbeta, order=2)

Return a list of thermal states with the specified betas
"""
function get_thermal_states(mps::AbstractMPS, hamGates, βs, dβ; order=2)
	Nβ = length(βs)
	d = mps
	mps = identityMPS(mps)
	canonicalize!(mps)
	mpss = Array{typeof(mps),1}(undef,Nβ)
	layers = prepare_layers(mps, hamGates,-dβ*1im/2, order)
	β=0
	βout = Float64[]
	for n in 1:Nβ
		Nsteps = floor((βs[n]-β)/dβ)
		count=0
		while β < βs[n]
			apply_layers_nonunitary!(mps,layers)
			canonicalize!(mps)
			β += dβ
			count+=1
			if mod(count,floor(Nsteps/10))==0
				print("-",string(count/Nsteps)[1:3],"-")
			end
		end
		println(": State ", n ,"/",Nβ, " done.")
		push!(βout,β)
		mpss[n] = copy(mps)
	end
	return mpss, βout
end

"""
	apply_local_op!(mps,op)

Apply the operator at every site
"""
function apply_local_op!(mps,op)
    N = length(mps.Γ)
	Γ = mps.Γ
	if mps.purification
		op = auxillerate(op)
	end
 	for n in 1:N
    	@tensor Γ[n][:] := Γ[n][-1,2,-3]*op[-2,2]
	end
end

"""
	apply_local_op!(mps,op, site)

Apply the operator at the site
"""
function apply_local_op!(mps,op,site::Integer)
    N = length(mps.Γ)
	Γ = mps.Γ
	if mps.purification
		op = auxillerate(op)
	end
    @tensor mps.Γ[site][:] := mps.Γ[site][-1,2,-3]*op[-2,2]
end
