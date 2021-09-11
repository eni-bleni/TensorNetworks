#TODO compress MPO https://arxiv.org/pdf/1611.02498.pdf

struct MPOsite{T<:Number} <: AbstractArray{T,4}
    data::Array{T,4}
    # ishermitian::Bool
    # isunitary::Bool
end
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
Base.eltype(site::MPOsite{T}) where {T} = T
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
Base.size(mpo::AbstractMPO) = size(data(mpo))
Base.length(mpo::MPO) = length(data(mpo))
Base.IndexStyle(::Type{<:MPOsite}) = IndexLinear()
Base.IndexStyle(::Type{<:AbstractMPO}) = IndexLinear()
Base.getindex(mpo::MPOsite, i::Integer) = mpo.data[i]
Base.getindex(mpo::AbstractMPO, i::Integer) = mpo.data[i]
#Base.setindex!(mpo::MPOsite, v, I::Vararg{Integer,4}) = (mpo.data[I] = v)
#Base.setindex!(mpo::AbstractMPO, v, I::Vararg{Integer,N}) where {N} = (mpo.data[I] = v)

operator_length(mpo::AbstractMPO) = length(mpo)
operator_length(mpo::MPOsite) = 1

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
            mpo[i-1] = reshape(1/2*U*diagm(S),size(mpo[i-1])[1],d,d,D)
            mpo[i] = reshape(2*V,D,d,d,size(mpo[i])[4])
        end
    end
    mpo[L] = permutedims(cat(1,permutedims(mpo1[L],[1,4,2,3]),permutedims(mpo2[L],[1,4,2,3])),[1,3,4,2])
    if tol>0
        @tensor tmp[-1,-2,-3,-4,-5,-6] := mpo[L-1][-1,-2,-3,1]*mpo[L][1,-4,-5,-6]
        tmp = reshape(tmp,size(mpo[L-1])[1]*d*d,d*d*size(mpo[L])[4])
        F = svd(tmp)
        U,S,V = truncate_svd(F,D,tol)
        mpo[L-1] = reshape(1/2*U*diagm(S),size(mpo[L-1])[1],d,d,D)
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
