#TODO compress MPO https://arxiv.org/pdf/1611.02498.pdf
abstract type AbstractMPOsite{T} <: AbstractArray{T,4} end

struct MPOsite{T<:Number} <: AbstractMPOsite{T}
    data::Array{T,4}
    # ishermitian::Bool
    # isunitary::Bool
end
Base.getindex(g::MPOsite, I::Vararg{Int,4}) = getindex(data(g),I...)


function MPOsite(op::Array{<:Number,2})
    sop = size(op)
    return MPOsite(reshape(op,1,sop[1],sop[2],1))
end
function MPOsite(op::Array{T,4}) where {T}
    return MPOsite{T}(op)
end
# function MPOsite(op::Array{<:Number,4})
#     return MPOsite(op, norm(op - conj(permutedims(op,[1,3,2,4]))) ≈ 0)
# end
data(site::MPOsite) = site.data
Base.eltype(::MPOsite{T}) where {T} = T
# LinearAlgebra.ishermitian(site::MPOsite) = site.ishermitian
# isunitary(site::MPOsite) = site.isunitary
reverse_direction(site::MPOsite) = MPOsite(permutedims(data(site),[4,2,3,1]))
Base.transpose(site::MPOsite) = MPOsite(permutedims(data(site),[1,3,2,4]))
Base.adjoint(site::MPOsite) = MPOsite(conj(permutedims(data(site),[1,3,2,4])))
Base.conj(site::MPOsite) = MPOsite(conj(data(site)))
Base.:+(site1::MPOsite,site2::MPOsite) = MPOsite(data(site1) .+ data(site2)) 

abstract type AbstractMPO{T<:Number} <: AbstractVector{MPOsite{T}} end

struct MPO{T<:Number} <: AbstractMPO{T}
    data::Vector{MPOsite{T}}
end

struct ScaledIdentityMPOsite{T} <: AbstractMPOsite{T}
    data::T
    function ScaledIdentityMPOsite(scaling::T) where {T<:Number}
        new{T}(scaling)
    end
end
data(site::ScaledIdentityMPOsite) = site.data
const IdentityMPOsite = ScaledIdentityMPOsite(true)

Base.length(mpo::ScaledIdentityMPOsite) = 1
function Base.size(::ScaledIdentityMPOsite, i::Integer)
    if i==1 || i==4
        return 1
    else 
        @error "Physical dimension of ScaledIdentityMPOsite is arbitrary"
    end
end
LinearAlgebra.ishermitian(mpo::ScaledIdentityMPOsite) = isreal(mpo.data)
isunitary(mpo::ScaledIdentityMPOsite) = mpo.data'*mpo.data ≈ 1
Base.:*(x::K, g::ScaledIdentityMPOsite) where {K<:Number} = ScaledIdentityMPOsite(x*data(g))
Base.:*(g::ScaledIdentityMPOsite, x::K) where {K<:Number} = ScaledIdentityMPOsite(x*data(g))
Base.:/(g::ScaledIdentityMPOsite, x::K) where {K<:Number} = inv(x)*g
auxillerate(mpo::ScaledIdentityMPOsite) = mpo
Base.show(io::IO, g::ScaledIdentityMPOsite) = print(io, ifelse(true == data(g), "",string(data(g),"*")), string("IdentityMPOsite"))
Base.show(io::IO, ::MIME"text/plain", g::ScaledIdentityMPOsite) = print(io, ifelse(true == data(g), "", string(data(g),"*")), string("IdentityMPOsite"))

reverse_direction(site::ScaledIdentityMPOsite) = site
Base.transpose(site::ScaledIdentityMPOsite) = site
Base.adjoint(site::ScaledIdentityMPOsite) = conj(site)
Base.conj(site::ScaledIdentityMPOsite) = ScaledIdentityMPOsite(conj(data(site)))
Base.:+(site1::ScaledIdentityMPOsite, site2::ScaledIdentityMPOsite) = ScaledIdentityMPOsite(data(site1) + data(site2)) 

Base.:(==)(mpo1::ScaledIdentityMPOsite, mpo2::ScaledIdentityMPOsite) = data(mpo1) == data(mpo2)

function Base.getindex(g::ScaledIdentityMPOsite, I::Vararg{Int,4}) 
    @assert I[1] == I[4] == 1 "Accessing out of bounds index on ScaledIdentityMPOsite "
    val = I[2] == I[3] ? 1 : 0
    return data(g)*val
end

struct ScaledIdentityMPO{T} <: AbstractMPO{T}
    data::T
    length::Int
    function ScaledIdentityMPO(scaling::T,n::Integer) where {T<:Number}
        new{T}(scaling,n)
    end
end
Base.IndexStyle(::Type{<:ScaledIdentityMPO}) = IndexLinear()
Base.getindex(g::ScaledIdentityMPO, i::Integer) = g.data^(1/length(g)) *IdentityMPOsite
data(g::ScaledIdentityMPO) = g.data

IdentityMPO(n) = ScaledIdentityMPO(true,n)
Base.length(mpo::ScaledIdentityMPO) = mpo.length
LinearAlgebra.ishermitian(mpo::ScaledIdentityMPO) = isreal(mpo.data)
isunitary(mpo::ScaledIdentityMPO) = mpo.data'*mpo.data ≈ 1
Base.:*(x::K, g::ScaledIdentityMPO) where {K<:Number} = ScaledIdentityMPO(x*data(g), length(g))
Base.:*(g::ScaledIdentityMPO, x::K) where {K<:Number} = ScaledIdentityMPO(x*data(g), length(g))
Base.:/(g::ScaledIdentityMPO, x::K) where {K<:Number} = inv(x)*g
Base.show(io::IO, g::ScaledIdentityMPO) = print(io, ifelse(true ==data(g), "",string(data(g),"*")), string("IdentityMPO of length ", length(g)))
Base.show(io::IO, ::MIME"text/plain", g::ScaledIdentityMPO) = print(io, ifelse(true == data(g), "", string(data(g),"*")), string("IdentityMPO of length ", length(g)))

Base.:(==)(mpo1::ScaledIdentityMPO, mpo2::ScaledIdentityMPO) = data(mpo1) == data(mpo2) && length(mpo1) == length(mpo2)
# struct HermitianMPO{T<:Number} <: AbstractMPO{T}
#     data::MPO{T}
#     function HermitianMPO(mpo::MPO{T}) where {T}
#         mpo = MPO([(m+ conj(permutedims(m,[1,3,2,4])))/2 for m in mpo.data])
#         new{T}(mpo)
#     end
#     function HermitianMPO(sites::Vector{MPOsite{T}}) where {T}
#         mpo = MPO([(site + conj(permutedims(site,[1,3,2,4])))/2 for site in sites])
#         new{T}(mpo)
#     end
# end

MPO(mpo::MPOsite) = MPO([mpo])
MPO(op::Array{<:Number,2}) = MPO(MPOsite(op))
MPO(ops::Vector{T}) where {T} = MPO(map(MPOsite,ops))
data(mpo::MPO) = mpo.data
# HermitianMPO(mpo::MPOsite) = HermitianMPO(MPO([mpo]))
# HermitianMPO(op::Array{T,2}) where {T<:Number} = HermitianMPO(MPOsite(op))
# HermitianMPO(ops::Array{Array{T,4},1}) where {T<:Number} = HermitianMPO(map(op->MPOsite(op),ops))

Base.size(mposite::MPOsite) = size(data(mposite))
Base.length(mpo::MPOsite) = 1
Base.size(mpo::AbstractMPO) = size(data(mpo))
Base.length(mpo::MPO) = length(data(mpo))
Base.IndexStyle(::Type{<:MPOsite}) = IndexLinear()
Base.IndexStyle(::Type{<:AbstractMPO}) = IndexLinear()
Base.getindex(mpo::MPOsite, i::Integer) = mpo.data[i]
Base.getindex(mpo::AbstractMPO, i::Integer) = mpo.data[i]
#Base.setindex!(mpo::MPOsite, v, I::Vararg{Integer,4}) = (mpo.data[I] = v)
#Base.setindex!(mpo::AbstractMPO, v, I::Vararg{Integer,N}) where {N} = (mpo.data[I] = v)


"""
	auxillerate(mpo)

Return tensor⨂Id_aux
"""
function auxillerate(mpo::MPOsite)
	sop = size(mpo)
	d = sop[2]
	idop = Matrix{eltype(mpo)}(I,d,d)
	@tensor tens[:] := idop[-3,-5]*mpo.data[-1,-2,-4,-6]
	return MPOsite(reshape(tens,sop[1],d^2,d^2,sop[4]))
end

auxillerate(mpo::MPO) = MPO(auxillerate.(mpo.data))
#auxillerate(mpo::HermitianMPO) = HermitianMPO(auxillerate.(mpo.data))

# %% Todo
"""
gives the mpo corresponding to a*mpo1 + b*mpo2.
"""
function addmpos(mpo1,mpo2,a,b,Dmax,tol=0) #FIXME
    L = length(mpo1)
    d = size(mpo1[1])[2]
    mpo = Array{Array{Complex{Float64}}}(L)
    mpo[1] = permutedims(cat(1,permutedims(a*mpo1[1],[4,1,2,3]),permutedims(b*mpo2[1],[4,1,2,3])),[2,3,4,1])
    for i = 2:L-1
        mpo[i] = permutedims([permutedims(mpo1[i],[1,4,2,3]) zeros(size(mpo1[i])[1],size(mpo2[i])[4],d,d); zeros(size(mpo2[i])[1],size(mpo1[i])[4],d,d) permutedims(mpo2[i],[1,4,2,3])],[1,3,4,2])
        if tol>0 || size(mpo1[i])[3]+size(mpo2[i])[3] > Dmax
            @tensor tmp[-1,-2,-3,-4,-5,-6] := mpo[i-1][-1,-2,-3,1]*mpo[i][1,-4,-5,-6]
            tmp = reshape(tmp,size(mpo[i-1])[1]*d*d,d*d*size(mpo[i])[4])
            F = svd(tmp)
            U,S,V = truncate_svd(F,Dmax,tol)
            mpo[i-1] = reshape(1/2*U*Diagonal(S),size(mpo[i-1])[1],d,d,D)
            mpo[i] = reshape(2*V,D,d,d,size(mpo[i])[4])
        end
    end
    mpo[L] = permutedims(cat(1,permutedims(mpo1[L],[1,4,2,3]),permutedims(mpo2[L],[1,4,2,3])),[1,3,4,2])
    if tol>0
        @tensor tmp[-1,-2,-3,-4,-5,-6] := mpo[L-1][-1,-2,-3,1]*mpo[L][1,-4,-5,-6]
        tmp = reshape(tmp,size(mpo[L-1])[1]*d*d,d*d*size(mpo[L])[4])
        F = svd(tmp)
        U,S,V = truncate_svd(F,D,tol)
        mpo[L-1] = reshape(1/2*U*Diagonal(S),size(mpo[L-1])[1],d,d,D)
        mpo[L] = reshape(2*V,D,d,d,size(mpo[L])[4])
    end
    return mpo
end



"""
```multiplyMPOs(mpo1,mpo2; c=true)```
"""
function multiplyMPOs(mpo1,mpo2; c=true) #FIXME
    L = length(mpo1)
    mpo = similar(mpo1)
    for j=1:L
        if c
            @tensor temp[:] := mpo1[j].data[-1,-3,1,-5] * conj(mpo2[j].data[-2,-4,1,-6])
        else
            @tensor temp[:] := mpo1[j].data[-1,-3,1,-5] * mpo2[j].data[-2,1,-4,-6]
        end
        s=size(temp)
        mpo[j] = MPOsite(reshape(temp,s[1]*s[2],s[3],s[4],s[5]*s[6]))
    end
    return MPO(mpo)
end


function Matrix(mpo::MPO)
    n = length(mpo)
    T = eltype(mpo[1])
    tens = SparseArray(ones(T,1,1,1))
    for site in mpo[1:n]
        dat = SparseArray(data(site))
        @tensor tens[out, newout, in,newin, right] := tens[out,in,c] * dat[c,newout,newin, right]
        st =size(tens)
        tens = SparseArray(reshape(tens, st[1]*st[2],st[3]*st[4],st[5]))
    end
    return tens[:,:,1]
end