Base.size(g::AbstractGate) = size(g.data)
Base.getindex(g::AbstractGate{T,N}, I::Vararg{Int,N}) where {T,N} = getindex(g.data,I...)
# Base.setindex!(g::AbstractGate{T,N}, v, I::Vararg{Int,N}) where {T,N} = setindex!(g.data, v, I...)
Base.length(::AbstractSquareGate{T,N}) where {T,N} = Int(N/2)
LinearAlgebra.ishermitian(gate::GenericSquareGate) = gate.ishermitian
LinearAlgebra.ishermitian(gate::ScaledIdentityGate) = gate.ishermitian
isunitary(gate::GenericSquareGate) = gate.isunitary
isunitary(mat::AbstractArray{<:Number,2}) = mat'*mat ≈ one(mat) && mat*mat' ≈ one(mat)
Base.complex(::Type{<:GenericSquareGate{T,N}}) where {T,N} = GenericSquareGate{complex(T),N}

Base.:*(x::K, g::ScaledIdentityGate) where {K<:Number} = ScaledIdentityGate(x*data(g), length(g))
Base.:*(g::ScaledIdentityGate, x::K) where {K<:Number} = ScaledIdentityGate(x*data(g),length(g))
Base.:*(x::K, g::GenericSquareGate) where {K<:Number} = GenericSquareGate(x*data(g))
Base.:*(g::GenericSquareGate, x::K) where {K<:Number} = GenericSquareGate(x*data(g))
Base.:/(g::GenericSquareGate, x::K) where {K<:Number} = inv(x)*g
Base.:/(g::ScaledIdentityGate, x::K) where {K<:Number} = inv(x)*g

function Base.:*(g1::GenericSquareGate{<:Any,N}, g2::GenericSquareGate{<:Any,N}) where {N}
	Gate(gate(Matrix(g1)*Matrix(g2), Int(N/2)))
end
Base.:*(g1::AbstractSquareGate{<:Any,N}, g2::ScaledIdentityGate{<:Any,N}) where {N} = g1*data(g2)
Base.:*(g2::ScaledIdentityGate{<:Any,N}, g1::AbstractSquareGate{<:Any,N}) where {N} = g1*data(g2)

""" 
	kron(g1::AbstractSquareGate, g2::AbstractSquareGate)

Order is consistent with Base.kron.
"""
function Base.kron(g1::AbstractSquareGate, g2::AbstractSquareGate)
	s1 = [size(g1)...]
	s2 = [size(g2)...]
	l1 = length(g1)
	l2 = length(g2)
	return Gate(reshape(kron(data(g1),data(g2)), [s1[1:l1]; s2[1:l2]; s1[l1+1:end]; s2[l2+1:end]]...))
end
function Base.kron(g1::ScaledIdentityGate, g2::ScaledIdentityGate)
	l1 = length(g1)
	l2 = length(g2)
	return data(g1)*data(g2)*IdentityGate(l1+l2)
end

function repeatedgate(g::AbstractSquareGate,n)
	gout = deepcopy(g)
	for k in 1:n-1
		gout = kron(g,gout)
	end
	return gout
end

Base.:+(g1::GenericSquareGate{K,N}, g2::GenericSquareGate{T,N}) where {T,K,N} = GenericSquareGate(data(g1)+ data(g2))
Base.:+(g1::ScaledIdentityGate{T,N}, g2::ScaledIdentityGate{K,N}) where {T,K,N} = ScaledIdentityGate(data(g1)+ data(g2),length(g1))
Base.:+(g1::ScaledIdentityGate{T,N}, g2::GenericSquareGate{K,N}) where {T,K,N} = data(g1)*one(data(g2))+ data(g2)
Base.:+(g1::GenericSquareGate{K,N}, g2::ScaledIdentityGate{T,N}) where {T,K,N} = data(g2)*one(data(g1))+ data(g1)

Base.exp(g::GenericSquareGate{T,N}) where {T,N} = GenericSquareGate(gate(exp(Matrix(g)), Int(N/2)))
Base.exp(g::ScaledIdentityGate{T,N}) where {T,N} = ScaledIdentityGate(exp(data(g)),length(g))


Base.adjoint(g::GenericSquareGate{T,N}) where {T,N} = GenericSquareGate(gate(Matrix(g)', Int(N/2)))

Base.adjoint(g::ScaledIdentityGate) = ScaledIdentityGate(data(g)',length(g))
Base.transpose(g::ScaledIdentityGate) = g

data(gate::GenericSquareGate) = gate.data
data(gate::ScaledIdentityGate) = gate.data

Base.convert(::Type{GenericSquareGate{T,N}}, g::GenericSquareGate{K,N}) where {T,K,N} = GenericSquareGate(convert.(T,g.data))
Base.convert(::Type{<:GenericSquareGate}, m::Matrix{<:Any}) = GenericSquareGate(m)

Base.permutedims(g::GenericSquareGate, perm) = GenericSquareGate(permutedims(g.data,perm))

LinearAlgebra.Hermitian(squareGate::AbstractSquareGate) = (squareGate + squareGate')/2

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
    reshape(data(g),D,D)
end

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
