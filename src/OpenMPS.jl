const DEFAULT_OPEN_DMAX = 20
const DEFAULT_OPEN_TOL = 1e-12
const DEFAULT_OPEN_NORMALIZATION = true
const DEFAULT_OPEN_TRUNCATION = TruncationArgs(
    DEFAULT_OPEN_DMAX,
    DEFAULT_OPEN_TOL,
    DEFAULT_OPEN_NORMALIZATION,
)

Base.firstindex(mps::OpenMPS) = 1
Base.lastindex(mps::OpenMPS) = length(mps.Γ)
Base.IndexStyle(::Type{<:OpenMPS}) = IndexLinear()
function Base.getindex(mps::OpenMPS, i::Integer) 
	LinkSite(mps.Γ[i], mps.Λ[i], mps.Λ[i+1], mps.purification)
end
function Base.setindex!(mps::OpenMPS{T}, v::LinkSite{K}, i::Integer) where {T,K}
	mps.Γ[i] = convert.(T, v.Γ)
	mps.Λ[i] = convert.(T, v.Λ1)
	mps.Λ[i+1] = convert.(T, v.Λ2)
end

#%% Constructors
function OpenMPS(
        Γ::Array{Array{T,3},1},
        Λ::Array{Array{T,1},1};
        truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
        purification = false,
        error = 0.0,
    ) where {T}
    OpenMPS{T}(Γ, Λ, purification, truncation, error)
end

function OpenMPS(
    Γ::Array{Array{T,3},1},
    Λ::Array{Array{T,1},1},
    mps::OpenMPS;
    error = 0.0,
) where {T}
    OpenMPS{T}(Γ, Λ, mps.purification, mps.truncation, mps.error + error)
end
function OpenMPS(
    M::Array{Array{T,3},1};
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
    purification = false,
) where {T}
    Γ, Λ, error = ΓΛ_from_M(M, truncation)
    OpenMPS{T}(Γ, Λ, purification, truncation, error)
end
function OpenMPS(mps::OrthOpenMPS{T}) where {T}
    Γ, Λ, error = ΓΛ_from_M(mps.Γ, mps.truncation)
    OpenMPS{T}(Γ, Λ, mps.purification, mps.truncation, error)
end

Base.copy(mps::OpenMPS{T}) where {T} = OpenMPS([copy(getfield(mps, k)) for k = 1:length(fieldnames(OpenMPS))]...) 

"""
    randomOpenMPS(datatype, N, d, D, pur=false, trunc)

Return a random mps
"""
function randomOpenMPS(T::DataType, N, d, D; purification = false,
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
)
    Ms = Array{Array{T,3},1}(undef, N)
    for i = 1:N
        Ms[i] = rand(T, i == 1 ? 1 : D, d, i == N ? 1 : D)
    end
    mps = OpenMPS(Ms, purification = purification, truncation = truncation)
    return mps
end


"""
    identityOpenMPS(datatype, N, d, trunc)

Return the identity density matrix as a purification
"""
function identityOpenMPS(T::DataType, N, d;
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
)
    Γ = Array{Array{T,3},1}(undef, N)
    Λ = Array{Array{T,1},1}(undef, N + 1)
    for i = 1:N
        Γ[i] = reshape(Matrix{T}(I, d, d) / sqrt(d), 1, d^2, 1)
        Λ[i] = ones(T, 1)
    end
    Λ[N+1] = ones(T, 1)
    return OpenMPS(Γ, Λ, purification = true, truncation = truncation)
end

"""
    identityOpenMPS(mps)

Return the identity density matrix as a purification, with similar parameters as the input mps
"""
function identityMPS(mps::OpenMPS{T}) where {T}
    N = length(mps.Γ)
    d = size(mps.Γ[1], 2)
    if mps.purification
        d = Int(sqrt(d))
    end
    trunc = mps.truncation
    return identityOpenMPS(T, N, d, truncation = trunc)
end

"""
    canonicalize(mps::OpenMPS)

Return the canonical version of the mps
"""
function canonicalize(mps::OpenMPS)
    M = centralize(mps, 1)
    canonicalizeM!(M)
    Γ, Λ, err = ΓΛ_from_M(M, mps.truncation)
    return OpenMPS(Γ, Λ, mps, error = err)
end

""" #TODO make consistent with OrthOpenMPS?
    canonicalize!(mps::OpenMPS)

Make the mps canonical
"""
function canonicalize!(mps::OpenMPS)
    M = centralize(mps, 1)
    canonicalizeM!(M)
    Γ, Λ, err = ΓΛ_from_M(M, mps.truncation)
    N = length(mps.Γ)
    mps.Γ = Γ
    mps.Λ = Λ
    mps.error += err
    return
end


"""
    iscanonical(mps::OpenMPS, eps)

Check if `mps` is canonical up to `eps`
"""
function iscanonical(mps::OpenMPS, eps = 1e-12)
    N = length(mps.Γ)
    err = 0.0
    for i in 1:N
        g = absorb_l(mps.Γ[i],mps.Λ[i],:left)
        sg = size(g)
        g = reshape(g,sg[1]*sg[2],sg[3])
        gg = g' * g
        println(norm((gg) .- Matrix(1.0I,size(gg))))
        err += norm((gg) .- Matrix(1.0I,size(gg)))

        g = absorb_l(mps.Γ[i],mps.Λ[i+1],:right)
        sg = size(g)
        g = reshape(g,sg[1],sg[2]*sg[3])
        gg = g*g'
        println(norm((gg) .- Matrix(1.0I,size(gg))))
        err += norm((gg) .- Matrix(1.0I,size(gg)))
    end
    err += abs(sum(map(s->sum(s.^2),mps.Λ) .- 1))
    return err<eps
end

"""
    centralize(mps::OpenMPS,n::Integer=1)

Contract Λ into Γ, such that site `n` is central. Returns the list of resulting tensors. If the mps is canonical, the output is leftcanonical up to `n` and right canonical afterwards.
"""
function centralize(mps::OpenMPS{T}, n::Integer = 1) where {T}
    N = length(mps.Γ)
    Γ = mps.Γ
    Λ = mps.Λ
    return M_from_ΓΛ(Γ, Λ, n)
end

function M_from_ΓΛ(Γ, Λ, n=1)
    N = length(Γ)
    M = copy(Γ)
    for k = N:-1:n+1
        absorb_l!(M[k], Γ[k], Λ[k+1])
    end
    for k = 1:n-1
        absorb_l!(M[k], Λ[k],Γ[k])
    end
    absorb_l!(M[n],Λ[n],Γ[n],Λ[n+1])
    return M
end


"""
    canonicalizeM(M::Array{Array{T,3},1}, n::Integer=1)

Make M left canonical up to site `n` and right canonical afterwards.
"""
function canonicalizeM(M_in::Array{Array{T,3},1}, n::Integer = 0) where {T}
    N = length(M)
    M = copy(M_in)
    for i = 1:n-1
        M[i], R, DB = LRcanonical(M[i], :left)
        if i < N
            @tensor M[i+1][:] := R[-1, 1] * M[i+1][1, -2, -3]
        end
    end
    for i = N:-1:n+1
        M[i], R, DB = LRcanonical(M[i], :right)
        if i > 1
            @tensor M[i-1][:] := M[i-1][-1, -2, 3] * R[3, -3]
        end
    end
    return M
end

"""
    canonicalizeM!(M::Array{Array{T,3},1}, n::Integer=1)

Make mps left canonical up to site `n` and right canonical afterwards.
"""
function canonicalizeM!(M::Array{Array{T,3},1}, n::Integer = 0) where {T}
    N = length(M)
    for i = 1:n-1
        M[i], R, DB = LRcanonical(M[i], :left)
        if i < N
            @tensor M[i+1][:] := R[-1, 1] * M[i+1][1, -2, -3]
        end
    end
    for i = N:-1:n+1
        M[i], R, DB = LRcanonical(M[i], :right)
        if i > 1
            @tensor M[i-1][:] := M[i-1][-1, -2, 3] * R[3, -3]
        end
    end
    return M
end


"""
    ΓΛ_from_M(M::Array{Array{T,3},1}; trunc::TruncationArgs)

Calculate Γ Λ from a list of tensors M. If M is right canonical, the result is canonical. THIS METHOD IS WRONG?
"""
function ΓΛ_from_M(M::Array{Array{T,3},1}, trunc::TruncationArgs) where {T}
    N = length(M)
    M = copy(M)
    Γ = copy(M)
    Λ = Array{Array{T,1},1}(undef, N + 1)
    Λ[1] = ones(size(M[1], 1))
    total_error = 0.0
    for k = 1:N
        st = size(M[k])
        tensor = reshape(M[k], st[1] * st[2], st[3])
        F = svd(tensor)
        U, S, Vt, D, err = truncate_svd(F, trunc)
        total_error += err
        Γ[k] = reshape(U, st[1], st[2], D)
        Γ[k] = absorb_l(Γ[k], 1 ./ Λ[k], :left)
        Λ[k+1] = S
        if k<N
            Vt = Diagonal(Λ[k+1]) * Vt
            @tensor M[k+1][:] := Vt[-1, 2] * M[k+1][2, -2, -3]
        end
    end
    # st = size(M[N])
    # q = qr(reshape(M[N], st[1] * st[2], st[3]))
    # Q = Matrix(q.Q)
    # Q = reshape(Q, st[1], st[2], size(Q, 2))
    # Q = absorb_l(Q, 1 ./ Λ[N], :left)
    # Γ[N] = Q #reshape(Q,st[1],st[2],size(q.R)[1])
    # Λ[N+1] = ones(size(Γ[N], 3))

    return Γ, Λ, total_error
end

#%% Expectation values
"""
    expectation_value(mps::OpenMPS, mpo::AbstractMPO, site::Integer)

Return the expectation value of the mpo starting at `site`

See also: [`expectation_values`](@ref), [`expectation_value_left`](@ref)
"""
function expectation_value(mps::OpenMPS, mpo::AbstractMPO, site::Integer = 1)
    oplength = operator_length(mpo)
    T = transfer_matrix(mps,mpo,site,:right)
    dl = size(mps.Γ[site],1)
    Λ2 = Diagonal(mps.Λ[site+oplength].^2)
    dr = length(mps.Λ[site+oplength])
    L = T*vec(Matrix(1.0I,dl,dl))
    return tr(Λ2*reshape(L,dr,dr))
end

"""
    expectation_value_left(mps::OpenMPS, mpo::AbstractMPO, site::Integer)

Return the expectation value of the mpo starting at `site`

See also: [`expectation_value`](@ref)
"""
function expectation_value_left(mps::OpenMPS, mpo::AbstractMPO, site::Integer)
    oplength = operator_length(mpo)
    T = transfer_matrix(mps,mpo,site,:left)
    dr = size(mps.Γ[site+oplength-1],3)
    Λ2 = Diagonal(mps.Λ[site].^2)
    R = T*vec(Matrix(1.0I,dr,dr))
    dl = length(mps.Λ[site])
    return tr(Λ2*reshape(R,dl,dl))
end

# %% TEBD
"""
    apply_layers!(mps::OpenMPS,layers)

Modify the mps by acting with the layers of gates
"""
function apply_layers!(mps::OpenMPS, layers)
    Γ = mps.Γ
    Λ = mps.Λ
    total_error = 0.0
    for n = 1:length(layers)
        Γ, Λ, err = apply_layer(Γ, Λ, layers[n], n, mps.truncation)
        total_error += err
    end
    mps.Γ = Γ
    mps.Λ = Λ
    mps.error += total_error
    return total_error
end

function apply_identity_layer!(mps::OpenMPS)
    Γ = mps.Γ
    Λ = mps.Λ
    total_error = 0.0
    for n = 1:2
        Γ, Λ, err = apply_identity_layer(Γ, Λ, n, mps.truncation)
        total_error += err
    end
    mps.Γ = Γ
    mps.Λ = Λ
    mps.error += total_error
    return total_error
end

"""
    apply_layers_nonunitary!(mps,layers)

Modify the mps by acting with the nonunitary layers of gates
"""
function apply_layers_nonunitary!(mps::OpenMPS, layers)
    Γ = mps.Γ
    Λ = mps.Λ
    total_error = 0.0
    for n = 1:length(layers)
        dir = isodd(n) ? 1 : -1
        total_error += apply_layer_nonunitary!(Γ, Λ, layers[n], n, dir,
            mps.truncation)
    end
    mps.error += total_error
    return total_error
end



"""
    norm(mps::OpenMPS)

Return the norm of the mps
"""
function LinearAlgebra.norm(mps::OpenMPS{T}) where {T}
    C = Array{T,2}(undef, 1, 1)
    C[1, 1] = one(T)
    N = length(mps.Γ)
    M = centralize(mps, N)
    for i = 1:N
        @tensor C[-1, -2] := M[i][2, 3, -2] * C[1, 2] * conj(M[i][1, 3, -1])
    end
    return C[1, 1]
end

# """
#     scalar_product(mps::OpenMPS, mps2::OpenMPS)

# Return the scalar product of the two mps's
# """
# function scalar_product(mps::OpenMPS{T}, mps2::OpenMPS{T}) where {T}
#     C = Array{T,2}(undef, 1, 1)
#     C[1, 1] = one(T)
#     N = length(mps.Γ)
#     M = centralize(mps, N)
#     M2 = centralize(mps2, N)
#     for i = 1:N
#         @tensor C[-1, -2] := M[i][2, 3, -2] * C[1, 2] * conj(M2[i][1, 3, -1])
#     end
#     return C[1, 1]
# end

"""
    scalar_product(mps::OpenMPS, mps2::OpenMPS)

Return the scalar product of the two mps's
"""
scalar_product(mps::OpenMPS{T}, mps2::OpenMPS{T}) where {T} = scalar_product(OrthOpenMPS(mps), OrthOpenMPS(mps2))
scalar_product(mps::OrthOpenMPS{T}, mps2::OpenMPS{T}) where {T} = scalar_product(mps, OrthOpenMPS(mps2))
scalar_product(mps::OpenMPS{T}, mps2::OrthOpenMPS{T}) where {T} = scalar_product(OrthOpenMPS(mps), mps2)

# %% Transfer 
transfer_matrix(mps::OpenMPS, mpo::MPOsite, site::Integer, direction=:left)	= transfer_matrix(mps[site], mpo, direction)

"""
    transfer_matrix(mps::OpenMPS, op, site, direction = :left)

Return the transfer matrix at `site` with the operator sandwiched
"""
function transfer_matrix(mps::OpenMPS, op::AbstractGate{T_op,N_op}, site::Integer, direction = :left) where {T_op, N_op}
	oplength = Int(length(size(op))/2)
	N = length(mps.Γ)
    if (site+oplength-1) >N
        error("Operator goes outside the chain.")
    end
    sites = mps[site:(site+oplength-1)]
    return transfer_matrix(sites,op,direction)
end

function saveOpenMPS(mps, filename)
    jldopen(filename, "w") do file
        writeOpenMPS(file, mps)
    end
end

function writeOpenMPS(parent, mps)
    write(parent, "Gamma", mps.Γ)
    write(parent, "Lambda", mps.Λ)
    write(parent, "Purification", mps.purification)
    write(parent, "Dmax", mps.truncation.Dmax)
    write(parent, "tol", mps.truncation.tol)
    write(parent, "normalize", mps.truncation.normalize)
    write(parent, "error", mps.error)
end

function readOpenMPS(io)
    Γ = read(io, "Gamma")
    Λ = read(io, "Lambda")
    purification = read(io, "Purification")
    Dmax = read(io, "Dmax")
    tol = read(io, "tol")
    normalize = read(io, "normalize")
    error = read(io, "error")
    trunc = TruncationArgs(Dmax, tol, normalize)
    mps = OpenMPS(
        Γ,
        Λ,
        purification = purification,
        truncation = trunc,
        error = error,
    )
    return mps
end

function loadOpenMPS(filename)
    jldopen(filename, "r") do file
        global mps
        mps = readOpenMPS(file)
    end
    return mps
end
