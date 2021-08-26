struct TruncationArgs
    Dmax::UInt8
    tol::Float64
	normalize::Bool
end
Base.copy(ta::TruncationArgs) = TruncationArgs([copy(getfield(ta, k)) for k = 1:length(fieldnames(TruncationArgs))]...) 

abstract type AbstractGate{T,N} <: AbstractArray{T,N} end
abstract type AbstractSquareGate{T,N} <: AbstractGate{T,N} end
struct GenericSquareGate{T,N} <: AbstractSquareGate{T,N}
    data::Array{T,N}
    function GenericSquareGate(data::AbstractArray{T,N}) where {T,N}
        @assert iseven(N) "Gate should be square"
        new{T,N}(data)
    end
end
struct HermitianGate{T,N} <: AbstractSquareGate{T,N}
    data::GenericSquareGate{T,N}
    function HermitianGate(data::AbstractArray{T,N}) where {T,N}
        hdata = gate(data |> GenericSquareGate |> Matrix |> Hermitian, Int(N/2))
        new{T,N}(GenericSquareGate(hdata))
    end
end
struct ScaledHermitianGate{T,N} <: AbstractSquareGate{T,N}
    data::HermitianGate{T,N}
    prefactor::Number
    function ScaledHermitianGate(gate::HermitianGate{T,N},prefactor=1) where {T,N}
        new{T,N}(gate,prefactor)
    end
end
struct UnitaryGate{T,N} <: AbstractSquareGate{T,N}
    data::GenericSquareGate{T,N}
    function UnitaryGate(data::AbstractArray{T,N}) where {T,N}
        mat = Matrix(GenericSquareGate(data)) 
        @assert (mat*mat' ≈ one(mat)) "Gate is not unitary"
        new{T,N}(GenericSquareGate(data))
    end
end

abstract type AbstractSite{} end
struct LinkSite{T<:Number} <: AbstractSite
    Γ::Array{T,3}
    Λ1::Array{T,1}
    Λ2::Array{T,1}
    purification::Bool 
end
Base.size(site::LinkSite{T}) where {T} = size(site.Γ)

struct GenericSite{T<:Number} <: AbstractSite
    Γ::Array{T,3}
    purification::Bool
end


abstract type AbstractMPS end
abstract type AbstractOpenMPS <: AbstractMPS end

mutable struct OpenMPS{T<:Complex} <: AbstractOpenMPS
    #In gamma-lambda notation
    Γ::Array{Array{T,3},1}
    Λ::Array{Array{T,1},1}

    #Indicates whether the MPS should be treated as a purification or not
    purification::Bool

    # Max bond dimension and tolerance
    truncation::TruncationArgs

    #Accumulated error
    error::Float64
end

mutable struct OrthOpenMPS{T<:Complex} <: AbstractOpenMPS
    Γ::Array{Array{T,3},1}

    #Indicates whether the MPS should be treated as a purification or not
    purification::Bool

    # Max bond dimension and tolerance
    truncation::TruncationArgs

    #Accumulated error
    error::Float64

    #Orthogonality boundaries
    lb::UInt8
    rb::UInt8
end

mutable struct UMPS{T <: Number} <: AbstractMPS
    #In gamma-lambda notation
    Γ::Array{Array{T,3},1}
    Λ::Array{Array{T,1},1}

    #Indicates whether the MPS should be treated as a purification or not
    purification::Bool

	# Max bond dimension and tolerance
	truncation::TruncationArgs

	# Accumulated error
	error::Float64
end

mutable struct CentralUMPS{T <: Number} <: AbstractMPS
    #In gamma-lambda notation
    ΓL::Vector{GenericSite{T}}
    ΓR::Vector{GenericSite{T}}
    Λ::Array{T,1}

    #Indicates whether the MPS should be treated as a purification or not
    purification::Bool

	# Max bond dimension and tolerance
	truncation::TruncationArgs

	# Accumulated error
	error::Float64
end