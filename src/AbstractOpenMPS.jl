isinfinite(::T) where {T<:BraOrKet} = isinfinite(boundaryconditions(T))
isinfinite(::OpenBoundary) = false
isinfinite(::InfiniteBoundary) = true

function boundary(::OpenBoundary, mps::BraOrKetOrVec, side)
    if side==:right
        return [one(eltype(mps[end]))]
    else 
        if side!==:left
            @warn "No direction chosen for the boundary vector. Defaulting to :left"
        end
        return [one(eltype(mps[1]))]
    end
end
boundary(bc::OpenBoundary, mps1::BraOrKetOrVec, mps2::BraOrKetOrVec,side) = kron(boundary(bc,mps1,side), boundary(bc,mps2,side))
boundary(bc::OpenBoundary, mps::BraOrKetOrVec, mpo::AbstractMPO,side) = boundary(bc,mps,side)
boundary(bc::OpenBoundary, mps1::BraOrKetOrVec, mpo::AbstractMPO, mps2::AbstractMPS, side) = kron(boundary(bc,mps1,side), boundary(bc,mps2,side))

boundary(mps::BraOrKet, args::Vararg) = boundary(boundaryconditions(mps),mps,args...)

function boundary(::InfiniteBoundary, mps::BraOrKet,g::ScaledIdentityMPO,mps2::BraOrKet, side::Symbol)
	_, rhos = transfer_spectrum(mps, mps2, reverse_direction(side),nev=1)
	return (data(g) ≈ 1 ? 1 : 0)*rhos[1]
end
function boundary(::InfiniteBoundary, mps::BraOrKet, side::Symbol)
	_, rhos = transfer_spectrum(mps, reverse_direction(side),nev=1)
	return canonicalize_eigenoperator(rhos[1])
end
function boundary(::InfiniteBoundary, mps::BraOrKet,mps2::BraOrKet, side::Symbol)
	_, rhos = transfer_spectrum(mps, mps2, reverse_direction(side),nev=1)
	return canonicalize_eigenoperator(rhos[1])
end

boundaryvec(args...) = copy(vec(boundary(args...)))


function expectation_value(mps::AbstractMPS{GenericSite}, op, site::Integer)
    mps = set_center(mps, site)
    return expectation_value(mps,op,site, iscanonical=true)
end

function apply_identity_layer(::OpenBoundary, mpsin::AbstractMPS{GenericSite}; kwargs...)
    truncation = get(kwargs, :truncation, mpsin.truncation)
    mps = set_center(mpsin,1)
    for k in 1:length(mps)-1
        A, S, B,  err = apply_two_site_gate(mps[k], mps[k+1], IdentityGate(2), truncation)
        mps.center += 1
        mps[k] = A
        mps[k+1] = S*B
        mps.error += err
    end
    return mps
end

function entanglement_entropy(mpsin::AbstractMPS{GenericSite}, link::Integer)
    N = length(mpsin)
    @assert 0<link<N
    mps = set_center(mpsin, link)
    _,S,_ = svd(mps[link],:leftorthogonal)
    return entropy(S)
end

