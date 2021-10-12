# Base.length(mps::AbstractMPS) = length(mps.Γ)
# Base.getindex(mps::AbstractMPS, I) = [mps[i] for i in I]
Base.IndexStyle(::Type{<:AbstractMPS}) = IndexLinear()
Base.size(mps::AbstractMPS) = size(mps.Γ)
# Base.firstindex(mps::AbstractMPS) = 1
# Base.lastindex(mps::AbstractMPS) = length(mps)
# Base.eltype(::Type{AbstractMPS{T}}) where {T}  = T
# Base.iterate(mps::AbstractMPS, state=1) = state > length(mps) ? nothing : (mps[state], state+1)
# Base.iterate(rS::Iterators.Reverse{<:AbstractMPS}, state=rS.itr.count) = state < 1 ? nothing : (mps[state], state-1)

ispurification(mps::AbstractMPS) = ispurification(mps[1])

Base.show(io::IO, mps::AbstractMPS) =
    print(io, "MPS: ", typeof(mps), "\nSites: ", eltype(mps) ,"\nLength: ", length(mps), "\nTruncation: ", mps.truncation)
Base.show(io::IO, m::MIME"text/plain", mps::AbstractMPS) = show(io,mps)

function scalar_product(mps1::BraOrKet, mps2::BraOrKet)
	K = numtype(mps1,mps2)
	Ts::Vector{LinearMap{K}} = transfer_matrices(mps1,mps2)
	vl = transfer_matrix_bond(mps1,mps2,1,:right)*boundaryvec(mps1,mps2,:right)
	vr::Vector{K} = boundaryvec(mps1,mps2,:right)
	for k in length(mps1):-1:1
		vr = Ts[k] * vr
	end
	return transpose(vr)*vl
end
# scalar_product(mps1::BraOrKet, mps2::BraOrKet) = scalar_product(mps1',mps2)
LinearAlgebra.norm(mps::BraOrKet) = scalar_product(mps',mps)

function prepare_layers(mps::AbstractMPS, gs::Vector{<:AbstractSquareGate}, dt, trotter_order)
	gates = ispurification(mps) ? auxillerate.(gs) : gs
	return prepare_layers(gates,dt,trotter_order)
end

"""
	get_thermal_states(mps, hamGates, betas, dbeta, order=2)

Return a list of thermal states with the specified betas
"""
function get_thermal_states(mps::AbstractMPS, hamGates, βs, dβ; order=2)
	Nβ = length(βs)
	mps = identityMPS(mps)
	canonicalize!(mps)
	mpss = Array{typeof(mps),1}(undef,Nβ)
	layers = prepare_layers(mps, hamGates,dβ*1im/2, order)
	β=0
	βout = Float64[]
	for n in 1:Nβ
		Nsteps = floor((βs[n]-β)/dβ)
		count=0
		while β < βs[n]
			mps = apply_layers_nonunitary(mps,layers)
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

function imaginaryTEBD(mps::AbstractMPS, hamGates, βtotal, dβ; order=2)
	mps = deepcopy(mps)
	canonicalize!(mps)
	layers = prepare_layers(mps, hamGates,dβ*1im, order)
	β=0
	count=0
	Nsteps = βtotal/dβ
	while β < βtotal
		mps = apply_layers_nonunitary(mps,layers)
		canonicalize!(mps)
		β += dβ
		count+=1
		if mod(count,floor(Nsteps/10))==0
			print("-",string(count/Nsteps)[1:3],"-")
		end
	end
	return mps
end


"""
	apply_local_op!(mps,op)

Apply the operator at every site
"""
function apply_local_op!(mps,op)
    N = length(mps.Γ)
	Γ = mps.Γ
	if ispurification(mps)
		op = auxillerate(op)
	end
 	for n in 1:N 
    	@tensor Γ[n][:] := Γ[n][-1,2,-3]*op[-2,2] #TODO Replace by Tullio or matrix mult?
	end
end

"""
	apply_local_op!(mps,op, site)

Apply the operator at the site
"""
function apply_local_op!(mps,op,site::Integer) #FIXME pass through to operation on site
    N = length(mps.Γ)
	Γ = mps.Γ
	if ispurification(mps)
		op = auxillerate(op)
	end
    @tensor mps.Γ[site][:] := mps.Γ[site][-1,2,-3]*op[-2,2] #TODO Replace by Tullio or matrix mult?
end
