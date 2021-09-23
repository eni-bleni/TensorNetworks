struct TruncationArgs
    Dmax::UInt16
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
    function ScaledIdentityGate(scaling::T,n::Integer) where {T<:Number}
        new{T,2*n}(scaling, isreal(scaling), scaling'*scaling ≈ 1)
    end
end
IdentityGate(n) = ScaledIdentityGate(true,n)

Base.show(io::IO, g::ScaledIdentityGate{T,N}) where {T,N} = print(io, ifelse(true ==data(g), "",string(data(g),"*")), string("IdentityGate of length ", Int(N/2)))
Base.show(io::IO, ::MIME"text/plain", g::ScaledIdentityGate) = print(io, ifelse(true == data(g), "", string(data(g),"*")), string("IdentityGate of length ", Int(N/2)))

struct GenericSquareGate{T,N} <: AbstractSquareGate{T,N}
    data::Array{T,N}
    ishermitian::Bool
    isunitary::Bool
    function GenericSquareGate(data::AbstractArray{T,N}) where {T,N}
        @assert iseven(N) "Gate should be square"
        sg = size(data)
        l = Int(N/2)
        D = *(sg[1:l]...)
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

abstract type AbstractSite end
abstract type AbstractOrthogonalSite <: AbstractSite end
abstract type AbstractVirtualSite <: AbstractSite end

struct ConjugateSite{T<:AbstractSite} <: AbstractSite
    site::T
end
struct LinkSite{T<:Number} <: AbstractVirtualSite 
    Λ::Vector{T}
end
struct VirtualSite{T<:Number} <: AbstractVirtualSite 
    Λ::Matrix{T}
end
struct GenericSite{T<:Number} <: AbstractSite
    Γ::Array{T,3}
    purification::Bool
end

struct OrthogonalLinkSite{T<:Number} <: AbstractOrthogonalSite
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

abstract type AbstractMPS end
abstract type AbstractOpenMPS <: AbstractMPS end

struct ConjugateMPS{T<:AbstractMPS} <: AbstractMPS
    mps::T
end
Base.IndexStyle(::Type{<:ConjugateMPS}) = IndexLinear()
Base.adjoint(mps::AbstractMPS) = ConjugateMPS(mps)
Base.adjoint(mps::ConjugateMPS) = mps.mps
Base.getindex(mps::ConjugateMPS, i::Integer) = getindex(mps.mps,i)'
Base.length(mps::ConjugateMPS) = length(mps.mps)
Base.lastindex(mps::ConjugateMPS) = lastindex(mps.mps)

mutable struct OpenMPS{T} <: AbstractOpenMPS
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

mutable struct LROpenMPS{T<:Number} <: AbstractOpenMPS
    Γ::Vector{AbstractOrthogonalSite}
    Λ::LinkSite{T}

    # Max bond dimension and tolerance
    truncation::TruncationArgs

    #Accumulated error
    error::Float64

    #Orthogonality boundaries
    center::UInt16

    function LROpenMPS(
        Γ::Vector{AbstractOrthogonalSite},
        Λ::LinkSite{T};
        truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
        center=1, error=0.0,
    ) where {T}
        N = length(Γ)
        @assert 0<center<=N+1 "Error in constructing LROpenMPS: center is not in the chain"
        @assert norm(data(Λ)) ≈ 1 "Error in constructing LROpenMPS: Singular values not normalized"
        new{T}(Γ, Λ, truncation, error, center)
    end
end

mutable struct LCROpenMPS{T<:Number} <: AbstractOpenMPS
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
        center = count
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

mutable struct UMPS{T <: Number} <: AbstractMPS
    #In gamma-lambda notation
    Γ::Vector{GenericSite{T}}
    Λ::Vector{LinkSite{T}}

	# Max bond dimension and tolerance
	truncation::TruncationArgs

	# Accumulated error
	error::Float64
end

mutable struct CentralUMPS{T<:Number} <: AbstractMPS
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
Base.eltype(::CentralUMPS{T}) where {T} = GenericSite{T}
Base.eltype(::UMPS{T}) where {T} = OrthogonalLinkSite{T}
Base.eltype(::LCROpenMPS{T}) where {T} = GenericSite{T}
Base.eltype(::LROpenMPS{T}) where {T} = GenericSite{T}
Base.eltype(::OpenMPS{T}) where {T} = OrthogonalLinkSite{T}

struct MPSSum
    states::Vector{Tuple{Number,AbstractMPS}}
end
Base.:+(mps1::Tuple{Number,AbstractMPS},mps2::Tuple{Number,AbstractMPS}) = MPSSum([mps1, mps2])
Base.:+(mps1::Tuple{Number,AbstractMPS},sum::MPSSum) = MPSSum(vcat([mps1], sum.states))
Base.:+(sum::MPSSum,mps1::Tuple{Number,AbstractMPS}) = MPSSum(vcat(sum.states, [mps1]))
Base.length(mps::MPSSum) = length(mps.states[1][2])