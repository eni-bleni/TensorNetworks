Base.getindex(mps::LCROpenMPS, i::Integer) = mps.Γ[i]
function Base.setindex!(mps::LCROpenMPS, v, i::Integer)
    c = center(mps)
    if i<c
        @assert isleftcanonical(v) "Error in setindex!: site not left canonical"
    elseif i==c 
        @assert norm(v) ≈ 1 "Error in setindex!: site not normalized, $(norm(v))"
    else
        @assert isrightcanonical(v) "Error in setindex!: site not right canonical"
    end
    mps.Γ[i] = v
end
Base.copy(mps::LCROpenMPS) = LCROpenMPS(copy(mps.Γ), truncation = copy(mps.truncation), error = copy(mps.error)) 
center(mps::LCROpenMPS) = mps.center

#TODO Iterative compression as in https://arxiv.org/abs/1008.3477 

#%% Constructors 
# function LCROpenMPS(
#     M::Vector{<:GenericSite};
#     truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
#     center=1, error = 0.0,
# )
#     Γ = to_left_right_orthogonal(M,center=center)
#     LCROpenMPS(Γ, truncation=truncation, error = error)
# end

function LCROpenMPS(
    M::Vector{Array{<:Number,3}};
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
    purification = false, center=1, error = 0.0,
)
    Γ = [GenericSite(m, purification) for m in M]
    Γ2 = to_left_right_orthogonal(Γ,center=center)
    LCROpenMPS(Γ2, truncation=truncation, error = error)
end

function LCROpenMPS(
    M::BraOrKetWith(OrthogonalLinkSite);
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
    center=1, error=0.0,
)
    N = length(M)
    Γ = Vector{GenericSite{eltype(M[1])}}(undef, N)
    for k in 1:N
        k<N && @assert data(M[k].Λ2) ≈ data(M[k+1].Λ1) "Error in constructing LCROpenMPS: Sites do not share links"
        if k<center
            Γ[k] = GenericSite(M[k],:left)
        end
        if k>center
            Γ[k] = GenericSite(M[k],:right)
        end
    end
    Γ[center] = M[center].Λ1 * M[center].Γ * M[center].Λ2
    LCROpenMPS(Γ, truncation=truncation, error = error)
end

# function LCROpenMPS(
#     mps::OpenMPS;
#     center::Int = 1,
#     error = 0.0,
# )
#     LCROpenMPS(mps[1:end], truncation=mps.truncation, error = mps.error + error, center=center)
# end

function shift_center_right!(mps::LCROpenMPS, method=:qr)
    c=center(mps)
    N=length(mps)
    if c==N+1
        @warn "Can't shift center right: Center is already at the end of the chain"
        return mps
    end
    mps.center += 1
    if c==0
        return mps
    end
    L,r = to_left_orthogonal(mps[c], method=method)
    mps[c] = L
    if c<N
        mps[c+1] = r*mps[c+1]
    else
        if !(abs.(data(r)) ≈ [one(eltype(data(r)))])
            @warn "Remainder is not 1 at the end of the chain: $r"
        end
    end
    #return LCROpenMPS(ΓL,Γ,ΓR,error=mps.error, truncation=mps.truncation)
    return mps
end

function shift_center_left!(mps::LCROpenMPS, method=:qr)
    c=center(mps)
    N = length(mps)
    if c==0
        @warn "Can't shift center left: Center is already at the beginning of the chain"
        return mps
    end
    mps.center -= 1
    if c==N+1
        return mps
    end
    R, l = to_right_orthogonal(mps[c], method=method)
    mps[c] = R
    if c>1
        mps[c-1] = mps[c-1]*l 
    else
        if !(abs.(data(l)) ≈ [one(eltype(data(l)))])
            @warn "Remainder is not 1 at the end of the chain: $l"
        end
    end
    #return LCROpenMPS(ΓL,Γ,ΓR,error=mps.error, truncation=mps.truncation)
    return mps
end

function shift_center!(mps::LCROpenMPS, n::Integer)
    if n>0
        f = shift_center_right!
    elseif n<0
        f = shift_center_left!
    else
        return mps
    end
    for k in 1:abs(n)
        f(mps)
    end
    return mps
end
function set_center!(mps::LCROpenMPS, n::Integer)
    c = center(mps)
    N = length(mps)
    if n<0||n>N+1
        @warn "Can't set the center outside of the chain"
    end
    dn = n-c
    return shift_center!(mps,dn)
end
function set_center(mps::LCROpenMPS, n::Integer)
    mps2 = copy(mps)
    set_center!(mps2,n)
    return mps2
end
iscenter(mps::LCROpenMPS, c::Integer) = c==center(mps)

# function qr_right(L::GenericSite{T}, R::GenericSite{T})
#     Q, r = to_left_orthogonal(L)
#     R2 = r*R
#     return Q, R2
# end

# function qr_left(L::GenericSite{T}, R::GenericSite{T})
#     Q, r = to_right_orthogonal(R)
#     L2 = L*r
#     return L2, Q
# end

"""
    randomLCROpenMPS(N, d, D; T=ComplexF64, purification=false, trunc)

Return a random mps
"""
function randomLCROpenMPS(N, d, D; T=ComplexF64, purification = false,
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
)
    Γ = Vector{GenericSite{T}}(undef, N)
    for i = 1:N
        χL = Int(round(d^(min(i-1, log(d,D), N+1-i))))
        χR = Int(round(d^(min(i, log(d,D), N-i))))
        Γ[i] = randomRightOrthogonalSite(χL, d, χR, T, purification = purification)
    end
    return LCROpenMPS(Γ, truncation = truncation)
end

"""
    identityLCROpenMPS(datatype, N, d, trunc)

Return the identity density matrix as a purification
"""
function identityLCROpenMPS(N, d; T=ComplexF64, 
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
) 
    Γ = Vector{GenericSite{T}}(undef, N)
    for i = 1:N
        Γ[i] = GenericSite(reshape(Matrix(one(T)I, d, d) / sqrt(d), 1, d^2, 1), true)
    end
    return LCROpenMPS(Γ, truncation = truncation)
end

"""
    identityMPS(mps::LCROpenMPS)

Return the identity density matrix as a purification, with similar parameters as the input mps
""" 
function identityMPS(mps::LCROpenMPS) #FIXME
    N = length(mps)
    d = size(mps[1], 2)
    if ispurification(mps)
        d = Int(sqrt(d))
    end
    trunc = mps.truncation
    return identityLCROpenMPS(N, d, truncation = trunc)
end


"""
    canonicalize(mps::LCROpenMPS, center=1)

Make mps left canonical up to the `center` site and right canonical after, assuming the input mps is canonical.
"""
function canonicalize(mps::LCROpenMPS; center=1)
    Γ = to_left_right_orthogonal(mps[1:length(mps)], center=center)
    LCROpenMPS(Γ, truncation = mps.truncation, error = mps.error)
end

function canonicalize!(mps::LCROpenMPS; center=1)
    Γ = to_left_right_orthogonal(mps[1:length(mps)], center=center)
    mps.Γ = Γ
    mps.center = center
end

entanglement_entropy(mps::LCROpenMPS) = entanglement_entropy(OpenMPS(mps))
