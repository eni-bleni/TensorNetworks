struct TruncationArgs
    Dmax::Int
    tol::Float64
	normalize::Bool
end
Base.copy(ta::TruncationArgs) = TruncationArgs([copy(getfield(ta, k)) for k = 1:length(fieldnames(TruncationArgs))]...) 

abstract type AbstractGate{T,N} <: AbstractArray{T,N} end
abstract type AbstractSquareGate{T,N} <: AbstractGate{T,N} end

struct ScaledIdentityGate{T,N} <: AbstractSquareGate{T,N}
    data::T
    ishermitian::Bool
    isunitary::Bool
    function ScaledIdentityGate(scaling::T,n::Integer) where {T}
        new{T,2*n}(scaling, isreal(scaling), scaling'*scaling ≈ 1)
    end
end
IdentityGate(n) = ScaledIdentityGate(true,n)

Base.show(io::IO, g::ScaledIdentityGate{T,N}) where {T,N} = print(io, ifelse(true ==data(g), "",string(data(g),"*")), string("IdentityGate of length ", Int(N/2)))
Base.show(io::IO, ::MIME"text/plain", g::ScaledIdentityGate{T,N}) where {T,N} = print(io, ifelse(true == data(g), "", string(data(g),"*")), string("IdentityGate of length ", Int(N/2)))

struct GenericSquareGate{T,N} <: AbstractSquareGate{T,N}
    data::Array{T,N}
    ishermitian::Bool
    isunitary::Bool
    function GenericSquareGate(data::AbstractArray{T,N}) where {T,N}
        @assert iseven(N) "Gate should be square"
        sg = size(data)
        l = Int(N/2)
        D = prod(sg[1:l])
        mat = reshape(data,D,D)
        new{T,N}(data, ishermitian(mat), isunitary(mat))
    end
end
# struct HermitianGate{T,N} <: AbstractSquareGate{T,N}
#     data::GenericSquareGate{T,N}
#     function HermitianGate(data::AbstractArray{T,N}) where {T,N}
#         hdata = gate(data |> GenericSquareGate |> Matrix |> Hermitian, Int(N/2))
#         new{T,N}(GenericSquareGate(hdata))
#     end
# end
# struct ScaledHermitianGate{T,N} <: AbstractSquareGate{T,N}
#     data::HermitianGate{T,N}
#     prefactor::Number
#     function ScaledHermitianGate(gate::HermitianGate{T,N},prefactor=1) where {T,N}
#         new{T,N}(gate,prefactor)
#     end
# end
# struct UnitaryGate{T,N} <: AbstractSquareGate{T,N}
#     data::GenericSquareGate{T,N}
#     function UnitaryGate(data::AbstractArray{T,N}) where {T,N}
#         mat = Matrix(GenericSquareGate(data)) 
#         @assert (mat*mat' ≈ one(mat)) "Gate is not unitary"
#         new{T,N}(GenericSquareGate(data))
#     end
# end

abstract type AbstractSite{T,N} <: AbstractArray{T,N} end
abstract type AbstractCenterSite{T} <: AbstractSite{T,3} end
# abstract type AbstractKetSite <: AbstractSite end
# abstract type AbstractBraSite{T<:AbstractKetSite} <: AbstractSite end

# abstract type AbstractStandardSite <: AbstractKetSite end
# abstract type AbstractOrthogonalSite <: AbstractKetSite end
abstract type AbstractVirtualSite{T} <: AbstractSite{T,2} end

struct AdjointSite{K,N,T<:AbstractSite} <:AbstractSite{T,N}
    parent::T
    function AdjointSite(A::AbstractSite{K,N}) where {K,N}
        new{K,N,typeof(A)}(A)
    end
end

struct LinkSite{T} <: AbstractVirtualSite{T}
    Λ::Diagonal{T,Vector{T}}
end
LinkSite(v::Vector) = LinkSite(Diagonal(v))
struct VirtualSite{T} <: AbstractVirtualSite{T}
    Λ::Matrix{T}
end
struct GenericSite{T} <: AbstractCenterSite{T}
    Γ::Array{T,3}
    purification::Bool
end

struct OrthogonalLinkSite{T} <: AbstractCenterSite{T}
    Γ::GenericSite{T}
    Λ1::LinkSite{T}
    Λ2::LinkSite{T}
    function OrthogonalLinkSite(Λ1::LinkSite, Γ::GenericSite{T}, Λ2::LinkSite; check=false) where {T}
        if check
            @assert isleftcanonical(Λ1 * Γ) "Error in constructing OrthogonalLinkSite: Is not left canonical"
            @assert isrightcanonical(Γ * Λ2) "Error in constructing OrthogonalLinkSite: Is not right canonical"
            @assert norm(Λ1) ≈ 1
            @assert norm(Λ2) ≈ 1
        end
        new{T}(Γ, Λ1, Λ2)
    end
end
# const AbstractMPS = AbstractVector{T} where {T<:AbstractSite}
abstract type AbstractMPS{T<:AbstractCenterSite} <: AbstractVector{T} end

Base.adjoint(::Type{Any}) = Any
Base.adjoint(site::AbstractSite) = AdjointSite(site)
Base.adjoint(T::Type{<:AbstractSite{<:Any,N}}) where {N} = AdjointSite{<:eltype(T), N, <:T}
# Base.adjoint(T::Type{<:AbstractSite}) = AdjointSite{eltype(T), N, T}
# myeltype(x) = eltype(x)
# myeltype(::Type{Any}) = Any
Base.adjoint(T::Type{<:AbstractMPS}) = Adjoint{<:adjoint(eltype(T)), <:T}
const BraOrKet = Union{AbstractMPS, Adjoint{<:Any,<:AbstractMPS}}
const BraOrKetOrVec = Union{AbstractVector{<:AbstractSite},Adjoint{<:Any,<:AbstractVector{<:AbstractSite}}}#Union{AbstractMPS, Adjoint{<:Any,<:AbstractMPS},AbstractVector{<:AbstractSite}, Adjoint{<:Any,AbstractVector{<:AbstractSite}}}

BraOrKetWith(T::Type{<:AbstractSite}) = Union{AbstractMPS{<:T}, Adjoint{<:Any,<:AbstractMPS{<:T}}}
BraOrKetLike(T::Type{<:Union{AbstractMPS,AbstractSite}}) = Union{<:T, <:adjoint(T)}

# abstract type AbstractKetMPS{T<:AbstractKetSite} <: AbstractMPS{T} end
# abstract type AbstractBraMPS{T<:AbstractBraSite} <: AbstractMPS{T} end
#abstract type AbstractBraMPS{T<:AbstractBraSite} <: AbstractVector{T} end
# abstract type AbstractOpenMPS <: AbstractMPS end

# struct ConjugateMPS{T<:AbstractMPS, S<:AbstractBraSite} <: AbstractMPS{S}
#     mps::T
#     function ConjugateMPS(mps::T) where {T<:AbstractMPS{<:AbstractKetSite}}
#         new{T, BraSite{eltype(mps)}}(mps)
#     end
# end
# Base.adjoint(mps::AbstractMPS) = ConjugateMPS(mps)
# Base.adjoint(mps::ConjugateMPS) = mps.mps
# Base.getindex(mps::ConjugateMPS, i::Integer) = getindex(mps.mps,i)'
# Base.length(mps::ConjugateMPS) = length(mps.mps)
# Base.lastindex(mps::ConjugateMPS) = lastindex(mps.mps)
# Base.adjoint(mps::Type{T}) where {T<:AbstractMPS} = ConjugateMPS{T,BraSite{eltype(mps)}}
boundaryconditions(::Type{<:Adjoint{<:Any,T}}) where {T<:AbstractMPS} = boundaryconditions(T)
# Base.size(mps::ConjugateMPS) = size(mps.mps)


# _braket(T::Type{<:AbstractKetSite}) = Union{T,AbstractBraSite{<:T}}
# _braket(T::Type{<:AbstractBraSite{K}}) where {K} = Union{T,K}

# _braket(T::Type{<:AbstractSite}) = Union{T,<:Adjoint{<:BraSite{<:eltype(T)}, <:T}}
# _braket(T::Type{<:AbstractMPS}) = Union{T,<:Adjoint{<:BraSite{<:eltype(T)}, <:T}}

# _braket(T::Type{<:AbstractMPS}) = Union{T,typeof(T)}
# _braket(T::Type{<:ConjugateMPS{M,K}}) where {M,K} = Union{T,M}


mutable struct OpenMPS{T} <: AbstractMPS{OrthogonalLinkSite{T}}
    #In gamma-lambda notation
    Γ::Vector{GenericSite{T}}
    Λ::Vector{LinkSite{T}}

    # Max bond dimension and tolerance
    truncation::TruncationArgs

    #Accumulated error
    error::Float64
    function OpenMPS(
        Γ::Vector{GenericSite{T}},
        Λ::Vector{LinkSite{T}};
        truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION, error=0.0) where {T}
        new{T}(Γ, Λ, truncation, error)
    end
end


mutable struct LCROpenMPS{T} <: AbstractMPS{GenericSite{T}}
    Γ::Vector{GenericSite{T}}

    # Max bond dimension and tolerance
    truncation::TruncationArgs

    #Accumulated error
    error::Float64

    center::Int16
    function LCROpenMPS(
        Γ::Vector{GenericSite{T}};
        truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
        error=0.0,
    ) where {T}
        count=1
        N = length(Γ)
        while count<N+1 && isleftcanonical(Γ[count]) 
            count+=1
        end
        center = min(count,N)
        if count<N+1
            if !(norm(data(Γ[count])) ≈ 1)
                @warn "LCROpenMPS is not normalized.\nnorm= $n"
            end
            count+=1
        end
        while count<N+1 && isrightcanonical(Γ[count])
            count+=1
        end
        @assert count == N+1 "LCROpenMPS is not LR canonical"
        new{T}(Γ, truncation, error, center)
    end
end

mutable struct UMPS{T} <: AbstractMPS{OrthogonalLinkSite{T}}
    #In gamma-lambda notation
    Γ::Vector{GenericSite{T}}
    Λ::Vector{LinkSite{T}}

	# Max bond dimension and tolerance
	truncation::TruncationArgs

	# Accumulated error
	error::Float64
end

mutable struct CentralUMPS{T} <: AbstractMPS{GenericSite{T}}
    #In gamma-lambda notation
    ΓL::Vector{GenericSite{T}}
    ΓR::Vector{GenericSite{T}}
    Λ::Vector{T}

    #Indicates whether the MPS should be treated as a purification or not
    purification::Bool

	# Max bond dimension and tolerance
	truncation::TruncationArgs

	# Accumulated error
	error::Float64
end

numtype(::LCROpenMPS{T}) where {T} = T
numtype(::UMPS{T}) where {T} = T
numtype(::CentralUMPS{T}) where {T} = T
numtype(::OpenMPS{T}) where {T} = T
numtype(m::Adjoint{<:Any,M}) where {M<:AbstractMPS} = numtype(m.parent)
numtype(ms::Vararg{BraOrKet,<:Any}) = promote_type(numtype.(ms)...)
struct MPSSum
    states::Vector{Tuple{Number,AbstractMPS}}
end
Base.:+(mps1::Tuple{Number,AbstractMPS},mps2::Tuple{Number,AbstractMPS}) = MPSSum([mps1, mps2])
Base.:+(mps1::Tuple{Number,AbstractMPS},sum::MPSSum) = MPSSum(vcat([mps1], sum.states))
Base.:+(sum::MPSSum,mps1::Tuple{Number,AbstractMPS}) = MPSSum(vcat(sum.states, [mps1]))
Base.length(mps::MPSSum) = length(mps.states[1][2])

abstract type BoundaryCondition end
struct OpenBoundary <: BoundaryCondition end
struct InfiniteBoundary <: BoundaryCondition end
boundaryconditions(::T) where {T<:BraOrKet} = boundaryconditions(T) 
boundaryconditions(::Type{<:OpenMPS}) = OpenBoundary()
boundaryconditions(::Type{<:LCROpenMPS}) = OpenBoundary()
boundaryconditions(::Type{<:UMPS}) = InfiniteBoundary()
# mutable struct LROpenMPS{T<:Number} <: AbstractOpenMPS
#     Γ::Vector{AbstractOrthogonalSite}
#     Λ::LinkSite{T}

#     # Max bond dimension and tolerance
#     truncation::TruncationArgs

#     #Accumulated error
#     error::Float64

#     #Orthogonality boundaries
#     center::Int

#     function LROpenMPS(
#         Γ::Vector{AbstractOrthogonalSite},
#         Λ::LinkSite{T};
#         truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
#         center=1, error=0.0,
#     ) where {T}
#         N = length(Γ)
#         @assert 0<center<=N+1 "Error in constructing LROpenMPS: center is not in the chain"
#         @assert norm(data(Λ)) ≈ 1 "Error in constructing LROpenMPS: Singular values not normalized"
#         new{T}(Γ, Λ, truncation, error, center)
#     end
# end
