Base.size(g::AbstractGate) = size(g.data)
Base.getindex(g::AbstractGate{T,N}, I::Vararg{Int,N}) where {T,N} = getindex(g.data,I...)
Base.setindex!(g::AbstractGate{T,N}, v, I::Vararg{Int,N}) where {T,N} = setindex!(g.data, v, I...)
operator_length(op::AbstractGate) = length(op)
Base.length(::AbstractSquareGate{T,N}) where {T,N} = Int(N/2)
Base.complex(::Type{<:GenericSquareGate{T,N}}) where {T,N} = GenericSquareGate{complex(T),N}
Base.complex(::Type{<:HermitianGate{T,N}}) where {T,N} = HermitianGate{complex(T),N}

#gateData(gate::AbstractGate) = AbstractGate.data
#GenericSquareGate{T,N}(data::AbstractArray{K,N}) where {K,T,N} = GenericSquareGate{T,N}(convert.(T,data))

Base.:*(x::K, g::GenericSquareGate) where {K<:Number} = GenericSquareGate(x*g.data)
Base.:*(g::GenericSquareGate, x::K) where {K<:Number} = GenericSquareGate(x*g.data)
Base.:*(x::K, g::HermitianGate) where {K<:Real} = HermitianGate(x*g.data)
Base.:*(g::HermitianGate, x::K) where {K<:Real} = HermitianGate(x*g.data)
Base.:*(x::K, g::HermitianGate) where {K<:Number} = ScaledHermitianGate(g,x)
Base.:*(g::HermitianGate, x::K) where {K<:Number} = ScaledHermitianGate(g,x)
Base.:*(x::K, g::ScaledHermitianGate) where {K<:Number} = ScaledHermitianGate(g.data, g.prefactor*x)
Base.:*(g::ScaledHermitianGate, x::K) where {K<:Number} = ScaledHermitianGate(g.data, g.prefactor*x)


function Base.exp(g::GenericSquareGate{T,N}) where {T,N}
    GenericSquareGate(gate(exp(Matrix(g)), Int(N/2)))
end
Base.exp(g::HermitianGate) = HermitianGate(exp(g.data))
function Base.exp(g::ScaledHermitianGate{T,N}) where {T,N} 
    if real(g.prefactor) ≈ 0
        return UnitaryGate(gate(exp(g.prefactor*Hermitian(Matrix(g.data))), Int(N/2)))
    else
        return exp(g.prefactor*g.data.data)
    end
end
    
Base.convert(::Type{GenericSquareGate{T,N}}, g::GenericSquareGate{K,N}) where {T,K,N} = GenericSquareGate(convert.(T,g.data))
Base.permutedims(g::GenericSquareGate, perm) = GenericSquareGate(permutedims(g.data,perm))
Base.convert(::Type{HermitianGate{T,N}},g::HermitianGate{T,N}) where {T,N} = g
Base.convert(::Type{HermitianGate{T,N}},g::HermitianGate{K,N}) where {T,K,N} = HermitianGate(convert(GenericSquareGate{T,N},g.data))
Base.permutedims(g::HermitianGate, perm) = HermitianGate(permutedims(g.data,perm))
LinearAlgebra.Hermitian(squareGate::GenericSquareGate) = HermitianGate(squareGate)
HermitianGate(squareGate::GenericSquareGate) = HermitianGate(squareGate.data)

function Base.Matrix(g::GenericSquareGate)
    sg = size(g)
    l = length(g)
    D = *(sg[1:l]...)
    reshape(g.data,D,D)
end
Base.Matrix(g::HermitianGate) = Matrix(g.data)
function gate(matrix::AbstractMatrix ,sites::Integer)
    sm = size(matrix)
    d = Int(sm[1]^(1/sites))
    reshape(matrix, repeat([d], 2*sites)...)
end
# function gate(matrix::AbstractMatrix; dim::Integer)
#     sm = size(matrix)
#     sites = Int(log(dim,sm[1]))
#     reshape(matrix, repeat([dim], sites))
# end

function transfer_matrix(sites::Vector{LinkSite{T}}, op::AbstractGate, direction=:left) where {T}
	if sites[1].purification
		op = auxillerate(op)
	end
	if direction==:left
		sites = map(site->GenericSite(site,:right),sites)
	elseif direction ==:right
		sites = map(site->GenericSite(site,:left),sites)
	end
	transfer_matrix(sites, op, direction)
end


function transfer_matrix(Γ::Vector{GenericSite{T}}, gate::AbstractSquareGate{T_op,N_op}, direction = :left) where {T, T_op, N_op}
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
transfer_right(Γ::Vector{GenericSite{T}}, gate::HermitianGate{T_op,N_op}) where {T,T_op,N_op} = transfer_right(Γ,gate.data)
function transfer_right(Γ::Vector{GenericSite{T}}, gate::GenericSquareGate{T_op,N_op}) where {T, T_op, N_op}
	op = gate.data
    oplength = Int(N_op/2)
	opsize = size(gate)
	opvec = reshape(permutedims(op,[(2*(1:oplength) .- 1)..., 2*(1:oplength)...]), *(opsize...))
	s_start = size(Γ[1])[1]
	s_final = size(Γ[oplength])[3]
	function T_on_vec(invec)
		v = reshape(invec,1,s_start,s_start)
		for k in 1:oplength
			@tensoropt (1,2) v[:] := conj(Γ[k].Γ[1,-2,-4])* v[-1,1,2]* Γ[k].Γ[2,-3,-5]
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

function expectation_value(sites::Vector{GenericSite}, gate::AbstractSquareGate{T,N}) where {T,N}
    @assert length(sites) == N
    transfer_matrix(sites,op,:left)
end
function expectation_value(sites::Vector{LinkSite{T}}, gate::AbstractSquareGate{K,N}) where {T,K,N}
    @assert length(sites) == length(gate)
    Λ = diagm(sites[1].Λ1 .^2)
    transfer = transfer_matrix(sites,gate,:left)
    idR = vec(Matrix{T}(I, size(Λ)))
    return vec(Λ)'*(transfer*idR)
end
function expectation_value(sites::Vector{GenericSite{T}}, gate::AbstractSquareGate{K,N}) where {T,K,N}
    @assert length(sites) == length(gate)
    transfer = transfer_matrix(sites,gate,:left)
    DL = size(sites[1],1)
    DR = size(sites[end],3)
    idL = vec(Matrix{T}(I,DL,DL))'
    idR = vec(Matrix{T}(I,DR,DR))
    return idL*(transfer*idR)
end