Base.size(mps::OrthOpenMPS) = size(mps.Γ)
Base.IndexStyle(::Type{<:OrthOpenMPS}) = IndexLinear()
Base.getindex(mps::OrthOpenMPS, i::Int) = mps.Γ[i]
Base.setindex!(mps::OrthOpenMPS, v, i::Int) = (mps.Γ[i] = v)
firstindex(mps::OrthOpenMPS) = 1
lastindex(mps::OrthOpenMPS) = length(mps.Γ)

#%% Constructors 
function OrthOpenMPS(
    M::Array{Array{T,3},1};
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
    purification = false,
) where {T}
    OrthOpenMPS{T}(M, purification, truncation, 0.0, 1, length(M))
end

function OrthOpenMPS(
    Γ::Array{Array{T,3},1},
    Λ::Array{Array{T,1},1};
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
    purification = false,
    error = 0.0,
) where {T}
    M = M_from_ΓΛ(Γ,Λ,1)
    OrthOpenMPS{T}(M, purification, truncation, error,1,1)
end

function OrthOpenMPS(
    Γ::Array{Array{T,3},1},
    mps::AbstractOpenMPS,
    lb,rb;
    error = 0.0,
) where {T}
    M = canonicalizeM(Γ)
    out = OrthOpenMPS{T}(M, mps.purification, mps.truncation, mps.error + error, lb, rb)
end

function OrthOpenMPS(
    mps::OpenMPS{T},
    lb=1::Int, rb=1::Int;
    error = 0.0,
) where {T}
    if lb>rb
        error("Orthogonality boundaries do not satisfy lb<rb")
    end
    Γ = centralize(mps, lb)
    OrthOpenMPS{T}(Γ, mps.purification, mps.truncation, mps.error + error, lb, rb)
end
OrthOpenMPS(mps::OrthOpenMPS) = mps

function OrthOpenMPS(
    mps::OpenMPS{T};
    center = 1,
    error = 0.0,
) where {T}
    OrthOpenMPS{T}(mps, center, center, error = error)
end



"""
    randomOrthOpenMPS(datatype, N, d, D, pur=false, trunc)

Return a random mps
"""
function randomOrthOpenMPS(T::DataType, N, d, D; purification = false,
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
)
    Ms = Array{Array{T,3},1}(undef, N)
    for i = 1:N
        Ms[i] = rand(T, i == 1 ? 1 : D, d, i == N ? 1 : D)
    end
    mps = OrthOpenMPS(Ms, purification = purification, truncation = truncation)
    return mps
end

"""
    identityOrthOpenMPS(datatype, N, d, trunc)

Return the identity density matrix as a purification
"""
function identityOrthOpenMPS(T::DataType, N, d;
    truncation::TruncationArgs = DEFAULT_OPEN_TRUNCATION,
)
    Γ = Array{Array{T,3},1}(undef, N)
    for i = 1:N
        Γ[i] = reshape(Matrix{T}(I, d, d) / sqrt(d), 1, d^2, 1)
    end
    return OrthOpenMPS(Γ, purification = true, truncation = truncation)
end

"""
    identityMPS(mps::OrthOpenMPS)

Return the identity density matrix as a purification, with similar parameters as the input mps
"""
function identityMPS(mps::OrthOpenMPS{T}) where {T}
    N = length(mps.Γ)
    d = size(mps.Γ[1], 2)
    if mps.purification
        d = Int(sqrt(d))
    end
    trunc = mps.truncation
    return identityOrthOpenMPS(T, N, d, truncation = trunc)
end


"""
    iscanonical(mps::OrthOpenMPS, eps)

Check if `mps` is canonical up to `eps`
"""
function iscanonical(mps::OrthOpenMPS, eps = 1e-12)
    N = length(mps.Γ)
    err = 0.0
    mps.center
    for i in 1:N
        g = mps.Γ[i]
        sg = size(g)
        if i<mps.lb
            g = reshape(g,sg[1]*sg[2],sg[3])
        elseif i>mps.rb
            g = reshape(g,sg[1],sg[2]*sg[3])
        elseif lb==i==rb
            g = reshape(g,sg[1]*sg[2]*sg[3],1)
        else
            warn("Several sites without orthogonality")
        end
        gg = g' * g
        println(norm((gg) .- Matrix(1.0I,size(gg))))
        err += norm((gg) .- Matrix(1.0I,size(gg)))
    end
    return err<eps
end

"""
    canonicalize!(mps::OrthOpenMPS, lb=0, rb=0; force=false)

Make mps left canonical up to site `lb` and right canonical after `rb`, assuming the input mps is canonical. Set force=true to ignore the canonicity of the input.
"""
function canonicalize!(mps::OrthOpenMPS, lb=0, rb=0; force=false)
    M = mps.Γ 
    N = length(M)
    il = force ?  1 : max(1, mps.lb)
    ir = force ?  N : min(N, mps.rb)
    for i = il:lb-1
        M[i], R, DB = LRcanonical(M[i], :left)
        if i < N
            @tensor M[i+1][:] := R[-1, 1] * M[i+1][1, -2, -3]
        end
    end
    for i = ir:-1:rb+1
        M[i], R, DB = LRcanonical(M[i], :right)
        if i > 1
            @tensor M[i-1][:] := M[i-1][-1, -2, 3] * R[3, -3]
        end
    end
    mps.lb=lb
    mps.rb=rb
end

"""
    canonicalize!(mps::OrthOpenMPS, center::Int=0)

Make mps left canonical up to site `center` and right canonical afterwards.
"""
centralize(mps::OrthOpenMPS, center::Int = 0) = canonicalize!(mps, center, center)


#%% Expectation values
""" #FIXME
    expectation_value(mps::OrthOpenMPS, op::Array{T_op,N_op}, site::Int)

Return the expectation value of the gate starting at the `site`
"""
function expectation_value(mps::OrthOpenMPS, op::Array{T_op,N_op}, site::Int) where {T_op <: Number, N_op}
    opLength = operator_length(op)
    if mps.purification
        op = auxillerate(op)
    end
    canonicalize!(mps,site)
    if opLength == 1
        val = expectation_value_one_site(mps.Γ[site], op)
    elseif opLength == 2
        val = expectation_value_two_site(mps.Γ[site:site+1], op)
    else
        error("Expectation value not implemented for operators of this size")
    end
    return val
end

"""
    expectation_value(mps::OrthOpenMPS, mpo::MPO, site::Int)

Return the expectation value of the mpo starting at `site`

See also: [`expectation_values`](@ref), [`expectation_value_left`](@ref)
"""
function expectation_value(mps::OrthOpenMPS, mpo::MPO, site::Int = 1)
    oplength = operator_length(mpo)
    T = transfer_matrix(mps,mpo,site,:right)
    dl = size(mps.Γ[site],1)
    dr = size(mps.Γ[site+oplength-1],3)
    L = T*vec(Matrix(1.0I,dl,dl))
    return tr(reshape(L,dr,dr))
end

"""
    norm(mps::OrthOpenMPS)

Return the norm of the mps
"""
function LinearAlgebra.norm(mps::OrthOpenMPS{T}) where {T}
    C = Array{T,2}(undef, 1, 1)
    C[1, 1] = one(T)
    N = length(mps.Γ)
    M = mps.Γ
    for i = 1:N
        @tensor C[-1, -2] := M[i][2, 3, -2] * C[1, 2] * conj(M[i][1, 3, -1])
    end
    return C[1, 1]
end

"""
    scalar_product(mps::OpenMPS, mps2::OpenMPS)

Return the scalar product of the two mps's
"""
function scalar_product(mps::OrthOpenMPS{T}, mps2::OrthOpenMPS{T}) where {T}
    C = Array{T,2}(undef, 1, 1)
    C[1, 1] = one(T)
    N = length(mps.Γ)
    M = mps.Γ
    M2 = mps2.Γ
    for i = 1:N
        @tensor C[-1, -2] := M[i][2, 3, -2] * C[1, 2] * conj(M2[i][1, 3, -1])
    end
    return C[1, 1]
end

"""
    transfer_matrix(mps::OpenMPS, op, site, direction = :left)

Return the transfer matrix at `site` with the operator sandwiched
"""
function transfer_matrix(mps::OrthOpenMPS, op::Array{T_op,N_op}, site::Integer, direction = :left) where {T_op, N_op}
	oplength = Int(length(size(op))/2)
	N = length(mps.Γ)
    if (site+oplength-1) >N
        error("Operator goes outside the chain.")
    end
    if mps.purification
        op = auxillerate(op)
    end
    Γ = mps.Γ[site:(site+oplength-1)]
    return transfer_matrix(Γ,op,direction)
end

# %% TEBD
"""
    apply_layers!(mps::OrthOpenMPS,layers)

Modify the mps by acting with the layers of gates
"""
function apply_layers!(mps::OrthOpenMPS, layers)
    omps = OpenMPS(mps)
    total_error = apply_layers!(omps, layers)
    mps2 = OrthOpenMPS(omps)
    mps.Γ = mps2.Γ
    mps.lb = mps2.lb
    mps.rb = mps2.rb
    return total_error
end


function apply_identity_layer!(mps::OrthOpenMPS)
    omps = OpenMPS(mps)
    total_error = apply_identity_layer!(omps)
    mps2 = OrthOpenMPS(omps)
    mps.Γ = mps2.Γ
    mps.lb = mps2.lb
    mps.rb = mps2.rb
    return total_error
end

"""
    apply_layers_nonunitary!(mps::OrthOpenMPS,layers)

Modify the mps by acting with the nonunitary layers of gates
"""
function apply_layers_nonunitary!(mps::OrthOpenMPS, layers)
    omps = OpenMPS(mps)
    total_error = apply_layers_nonunitary!(omps, layers)
    mps2 = OrthOpenMPS(omps)
    mps.Γ = mps2.Γ
    mps.lb = mps2.lb
    mps.rb = mps2.rb
    return total_error
end