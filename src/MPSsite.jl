operator_length(op::AbstractSite) = 1

Base.permutedims(site::GenericSite, perm) = GenericSite(permutedims(site.Γ,perm), site.purification)
Base.copy(site::LinkSite{T}) where {T} = LinkSite([copy(getfield(site, k)) for k = 1:length(fieldnames(LinkSite))]...) 
Base.copy(site::GenericSite{T}) where {T} = GenericSite([copy(getfield(site, k)) for k = 1:length(fieldnames(GenericSite))]...) 
Base.size(site::AbstractSite) = size(site.Γ)
Base.size(site::AbstractSite, dim) = size(site.Γ, dim)

function transfer_matrix(mps::LinkSite, mpo::MPOsite, direction=:left)
    if mps.purification
		mpo = auxillerate(mpo)
	end
	if direction == :left
		T = transfer_left(mps.Γ, mps.Λ2, mpo)
	elseif direction == :right
		T = transfer_right(mps.Γ, mps.Λ1, mpo)
	else
		error("Choose direction :left or :right")
	end
	return T
end

function transfer_matrix(mps::GenericSite{K}, direction=:left) where {K}
	if direction == :left
		T = transfer_left(mps.Γ)
	elseif direction == :right
		T = transfer_right(mps.Γ)
	else
		error("Choose direction :left or :right")
	end
	return T
end

function transfer_matrix(mps::GenericSite, mpo::MPOsite, direction=:left)
    if mps.purification
		mpo = auxillerate(mpo)
	end
	if direction == :left
		T = transfer_left(mps.Γ, mpo)
	elseif direction == :right
		T = transfer_right(mps.Γ, mpo)
	else
		error("Choose direction :left or :right")
	end
	return T
end

function GenericSite(site::LinkSite{T}, direction = :left) where {T}
	if direction==:left
		return GenericSite{T}(absorb_l(site.Γ, site.Λ1, :left), site.purification)
	elseif direction==:right
		return GenericSite{T}(absorb_l(site.Γ, site.Λ2, :right), site.purification)
	end
end

# function absorb_on(site::LinkSite{T}, direction = :left) where {T}
# 	if direction==:left
# 		return GenericSite{T}(absorb_l(site.Γ, site.Λ1, :left), site.purification)
# 	elseif direction==:right
# 		return GenericSite{T}(absorb_l(site.Γ, site.Λ2, :right), site.purification)
# 	end
# end
absorb(site::LinkSite{T}) where {T} = GenericSite{T}(absorb_l(site.Λ1, site.Γ, site.Λ2), site.purification)

expectation_value(site::LinkSite, mpo::MPOsite) = expectation_value(absorb(site),mpo)

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

function auxillerate(gate::HermitianGate{T,N}) where {T,N}
	HermitianGate(auxillerate(gate.data))
end
