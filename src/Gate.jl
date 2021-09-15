Base.size(g::AbstractGate) = size(g.data)
Base.getindex(g::AbstractGate{T,N}, I::Vararg{Int,N}) where {T,N} = getindex(g.data,I...)
# Base.setindex!(g::AbstractGate{T,N}, v, I::Vararg{Int,N}) where {T,N} = setindex!(g.data, v, I...)
operator_length(op::AbstractGate) = length(op)
Base.length(::AbstractSquareGate{T,N}) where {T,N} = Int(N/2)
LinearAlgebra.ishermitian(gate::GenericSquareGate) = gate.ishermitian
isunitary(gate::GenericSquareGate) = gate.isunitary
isunitary(mat::AbstractArray{<:Number,2}) = mat'*mat ≈ one(mat) && mat*mat' ≈ one(mat)
Base.complex(::Type{<:GenericSquareGate{T,N}}) where {T,N} = GenericSquareGate{complex(T),N}
# Base.complex(::Type{<:HermitianGate{T,N}}) where {T,N} = HermitianGate{complex(T),N}

#gateData(gate::AbstractGate) = AbstractGate.data
#GenericSquareGate{T,N}(data::AbstractArray{K,N}) where {K,T,N} = GenericSquareGate{T,N}(convert.(T,data))

Base.:*(x::K, g::ScaledIdentityGate) where {K<:Number} = ScaledIdentityGate(x*data(g))
Base.:*(g::ScaledIdentityGate, x::K) where {K<:Number} = ScaledIdentityGate(x*data(g))
Base.:*(x::K, g::GenericSquareGate) where {K<:Number} = GenericSquareGate(x*data(g))
Base.:*(g::GenericSquareGate, x::K) where {K<:Number} = GenericSquareGate(x*data(g))

function Base.:*(g1::GenericSquareGate{T,N},g2::GenericSquareGate{K,N}) where {T,K,N}
	Gate(gate(Matrix(g1)*Matrix(g2), Int(N/2)))
end

Base.:+(g1::GenericSquareGate, g2::GenericSquareGate) = GenericSquareGate(data(g1)+ data(g2))
Base.:+(g1::ScaledIdentityGate, g2::ScaledIdentityGate) = ScaledIdentityGate(data(g1)+ data(g2))

Base.exp(g::GenericSquareGate{T,N}) where {T,N} = GenericSquareGate(gate(exp(Matrix(g)), Int(N/2)))
Base.exp(g::ScaledIdentityGate) = ScaledIdentityGate(exp(data(g)))


Base.adjoint(g::GenericSquareGate{T,N}) where {T,N} = GenericSquareGate(gate(Matrix(g)', Int(N/2)))

Base.adjoint(g::ScaledIdentityGate) = ScaledIdentityGate(data(g)')
Base.transpose(g::ScaledIdentityGate) = g
# Base.:*(x::K, g::HermitianGate) where {K<:Real} = HermitianGate(x*g.data)
# Base.:*(g::HermitianGate, x::K) where {K<:Real} = HermitianGate(x*g.data)
# Base.:*(x::K, g::HermitianGate) where {K<:Number} = ScaledHermitianGate(g,x)
# Base.:*(g::HermitianGate, x::K) where {K<:Number} = ScaledHermitianGate(g,x)
# Base.:*(x::K, g::ScaledHermitianGate) where {K<:Number} = ScaledHermitianGate(g.data, g.prefactor*x)
# Base.:*(g::ScaledHermitianGate, x::K) where {K<:Number} = ScaledHermitianGate(g.data, g.prefactor*x)

data(gate::GenericSquareGate) = gate.data
data(gate::ScaledIdentityGate) = gate.data

# Base.exp(g::HermitianGate) = HermitianGate(exp(g.data))
# function Base.exp(g::ScaledHermitianGate{T,N}) where {T,N} 
#     if real(g.prefactor) ≈ 0
#         return UnitaryGate(gate(exp(g.prefactor*Hermitian(Matrix(g.data))), Int(N/2)))
#     else
#         return exp(g.prefactor*g.data.data)
#     end
# end
    
Base.convert(::Type{GenericSquareGate{T,N}}, g::GenericSquareGate{K,N}) where {T,K,N} = GenericSquareGate(convert.(T,g.data))
Base.permutedims(g::GenericSquareGate, perm) = GenericSquareGate(permutedims(g.data,perm))
# Base.convert(::Type{HermitianGate{T,N}},g::HermitianGate{T,N}) where {T,N} = g
# Base.convert(::Type{HermitianGate{T,N}},g::HermitianGate{K,N}) where {T,K,N} = HermitianGate(convert(GenericSquareGate{T,N},g.data))
# Base.permutedims(g::HermitianGate, perm) = HermitianGate(permutedims(g.data,perm))
LinearAlgebra.Hermitian(squareGate::AbstractSquareGate) = (squareGate + squareGate')/2
# HermitianGate(squareGate::GenericSquareGate) = HermitianGate(squareGate.data)

function Gate(data::Array{T,N}) where {T<:Number,N}
	if iseven(N)
		return GenericSquareGate(data)
	else
		error("No gate with $N legs implemented")
		return GenericSquareGate(data)
	end
end
function Base.Matrix(g::GenericSquareGate)
    sg = size(g)
    l = length(g)
    D = *(sg[1:l]...)
    reshape(g.data,D,D)
end
# Base.Matrix(g::HermitianGate) = Matrix(g.data)
function gate(matrix::AbstractMatrix ,sites::Integer)
    sm = size(matrix)
    d = Int(sm[1]^(1/sites))
    reshape(matrix, repeat([d], 2*sites)...)
end

"""
	auxillerate(gate::AbstractSquareGate{T,N})

Return gate_phys⨂Id_aux
"""
function auxillerate(op::GenericSquareGate{T,N}) where {T,N}
	opSize = size(op)
	d::Int = opSize[1]
	opLength = Int(N/2)
	idop = reshape(Matrix{T}(I,d^opLength,d^opLength),opSize...)
	odds = -1:-2:(-4*opLength)
	evens = -2:-2:(-4*opLength)
	tens::Array{T,2*N} = ncon((op.data,idop),(odds,evens))
	return GenericSquareGate(reshape(tens,(opSize .^2)...))
end
auxillerate(gate::ScaledIdentityGate) = gate
# function gate(matrix::AbstractMatrix; dim::Integer)
#     sm = size(matrix)
#     sites = Int(log(dim,sm[1]))
#     reshape(matrix, repeat([dim], sites))
# end

function transfer_matrix(sites::Vector{<:OrthogonalLinkSite}, op::AbstractGate, direction=:left)
	if ispurification(sites[1])
		op = auxillerate(op)
	end
	if direction==:left
		sites = map(site->GenericSite(site,:right),sites)
	elseif direction ==:right
		sites = map(site->GenericSite(site,:left),sites)
	end
	transfer_matrix(sites, op, direction)
end

transfer_matrix(Γ::Vector{<:GenericSite}, g::ScaledIdentityGate, direction = :left) = data(g)*transfer_matrix(Γ,direction)

function transfer_matrix(Γ::Vector{<:GenericSite}, gate::AbstractSquareGate{T_op,N_op}, direction = :left) where {T_op, N_op}
    #op = gate.data
    oplength = Int(N_op/2)
	#opsize = size(op)
	if direction == :left
		Γnew = copy(reverse(Γ))
		for k = 1:oplength
			 Γnew[oplength+1-k] = permutedims(Γnew[oplength+1-k], [3,2,1])
			 gate = permutedims(gate,[oplength:-1:1..., 2*oplength:-1:oplength+1...])
		end
	elseif direction == :right
		Γnew = copy(Γ)
	else
		error("Specify :left or :right in transfer matrix calculation")
	end
    return transfer_right(Γnew, gate)
end
# transfer_right(Γ::Vector{GenericSite{T}}, gate::HermitianGate{T_op,N_op}) where {T,T_op,N_op} = transfer_right(Γ,gate.data)
function transfer_right(Γ::Vector{<:GenericSite}, gate::GenericSquareGate{T_op,N_op}) where {T_op, N_op}
	op = gate.data
    oplength = Int(N_op/2)
	opsize = size(gate)
	opvec = reshape(permutedims(op,[(2*(1:oplength) .- 1)..., 2*(1:oplength)...]), *(opsize...))
	s_start = size(Γ[1])[1]
	s_final = size(Γ[oplength])[3]
	function T_on_vec(invec)
		v = reshape(invec,1,s_start,s_start)
		for k in 1:oplength
			@tensoropt (1,2) v[:] := conj(data(Γ[k])[1,-2,-4])* v[-1,1,2]* data(Γ[k])[2,-3,-5]
			sv = size(v)
			v = reshape(v,*(sv[1:3]...),sv[4],sv[5])
		end
		@tensor v[:] := v[1,-1,-2] * opvec[1]
		#v = reshape(ncon((v,op),[[1:2*oplength...,-1,-2],1:2*oplength]),s_final^2)
		return vec(v)
	end
	return LinearMap{ComplexF64}(T_on_vec,s_final^2,s_start^2)
end
# function transfer_matrix(Γ::Vector{GenericSite{T}}, gate::AbstractGate{T_op,N_op}, direction = :left) where {T, T_op, N_op}
# 	op = gate.data
#     oplength = Int(N_op/2)
# 	opsize = size(op)
# 	if direction == :left
# 		Γnew = copy(reverse(Γ))
# 		for k = 1:oplength
# 			 Γnew[oplength+1-k] = permutedims(Γnew[oplength+1-k], [3,2,1])
# 			 op = permutedims(op,[oplength:-1:1..., 2*oplength:-1:oplength+1...])
# 		end
# 	elseif direction == :right
# 		Γnew = copy(Γ)
# 	else
# 		error("Specify :left or :right in transfer matrix calculation")
# 	end

# 	#println([(2*(1:oplength) .- 1)..., 2*(1:oplength)...])
# 	op = reshape(permutedims(op,[(2*(1:oplength) .- 1)..., 2*(1:oplength)...]), *(opsize...))
# 	s_start::Integer = size(Γnew[1])[1]
# 	s_final::Integer = size(Γnew[oplength])[3]

# 	function T_on_vec(vec)
# 		v = reshape(vec,1,s_start,s_start)
# 		for k in 1:oplength
# 			@tensoropt (1,2) v[:] := conj(Γnew[k].Γ[1,-2,-4])* v[-1,1,2]* Γnew[k].Γ[2,-3,-5]
# 			sv = size(v)
# 			v = reshape(v,*(sv[1:3]...),sv[4],sv[5])
# 		end
# 		@tensor v[:] := v[1,-1,-2] * op[1]
# 		#v = reshape(ncon((v,op),[[1:2*oplength...,-1,-2],1:2*oplength]),s_final^2)
# 		return reshape(v,s_final^2)
# 	end
# 	return LinearMap{ComplexF64}(T_on_vec,s_final^2,s_start^2)
# end

"""
    expectation_value(mps::AbstractMPS, op::AbstractGate, site::Integer)

Return the expectation value of the gate starting at the `site`
"""
function expectation_value(mps::AbstractMPS, op::AbstractGate, site::Integer)
    opLength = operator_length(op)
    return expectation_value(mps[site:site+opLength-1], op)
end
# expectation_value(sites::AbstractMPS, ::IdentityGate) = expectation_value(mps[site:site+opLength-1], Ide)

# function expectation_value(sites::Vector{GenericSite}, gate::AbstractSquareGate{T,N}) where {T,N}
#     @assert length(sites) == N "Error in 'expectation value': length(sites) != length(gate)"
#     transfer_matrix(sites,gate,:left)
# end
function expectation_value(sites::Vector{OrthogonalLinkSite{T}}, gate::AbstractSquareGate) where {T}
    @assert length(sites) == length(gate)
    Λ = diagm(data(sites[1].Λ1) .^2)
    transfer = transfer_matrix(sites,gate,:left)
	DR = size(sites[end],3)
    idR = vec(Matrix{T}(I,DR,DR ))
    return vec(Λ)'*(transfer*idR)
end
function expectation_value(sites::Vector{GenericSite{T}}, gate::AbstractSquareGate) where {T}
    @assert length(sites) == length(gate) "Error in 'expectation value': length(sites) != length(gate)"
    transfer = transfer_matrix(sites,gate,:left)
    DL = size(sites[1],1)
    DR = size(sites[end],3)
    idL = vec(Matrix{T}(I,DL,DL))'
    idR = vec(Matrix{T}(I,DR,DR))
    return idL*(transfer*idR)
end
function expectation_value(sites::Vector{OrthogonalLinkSite{T}}, g::ScaledIdentityGate) where {T}
    Λ = diagm(data(sites[1].Λ1) .^2)
    transfer = transfer_matrix(sites,:left)
	DR = size(sites[end],3)
    idR = vec(Matrix{T}(I,DR,DR ))
    return data(g) * (vec(Λ)'*(transfer*idR))
end
function expectation_value(sites::Vector{GenericSite{T}}, g::ScaledIdentityGate) where {T}
	transfer = transfer_matrix(sites,:left)
    DL = size(sites[1],1)
    DR = size(sites[end],3)
    idL = vec(Matrix{T}(I,DL,DL))'
    idR = vec(Matrix{T}(I,DR,DR))
    return data(g) * (idL*(transfer*idR))
end

# expectation_value(sites::Vector{<:GenericSite}, gate::HermitianGate) = real.(expectation_value(sites, gate.data))
# expectation_value(sites::Vector{<:AbstractOrthogonalSite}, gate::AbstractSquareGate) = expectation_value(GenericSite.(sites), gate)