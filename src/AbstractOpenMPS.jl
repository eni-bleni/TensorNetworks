isinfinite(::AbstractOpenMPS) = false
isinfinite(::ConjugateMPS{<:AbstractOpenMPS}) = false
"""
    canonicalize(mps::AbstractOpenMPS, n)

Return the mps acted on with `n` layers of the identity
"""
function canonicalize(mps::AbstractOpenMPS,n)
    mps2 = copy(mps)
    for i in 1:n
        apply_identity_layer!(mps2)
    end
    return mps2
end
"""
    canonicalize!(mps::AbstractOpenMPS, n)

Act with `n` layers of the identity on `mps`
"""
function canonicalize!(mps::AbstractOpenMPS,n)
    for i in 1:n
        apply_identity_layer!(mps)
    end
end
function boundary(mps::Union{AbstractOpenMPS,ConjugateMPS{<:AbstractOpenMPS}}, side)
    if side==:left
        [one(eltype(mps[1]))]
    elseif side==:right
        [one(eltype(mps[end]))]
    end
end
boundary(mps1::ConjugateMPS{<:AbstractOpenMPS},mps2::AbstractOpenMPS,side) = kron(boundary(mps1,side), boundary(mps2,side))
boundary(mps::AbstractOpenMPS, mpo::AbstractMPO,side) = boundary(mps,side)
boundary(mps1::ConjugateMPS{<:AbstractOpenMPS},mpo::AbstractMPO, mps2::AbstractOpenMPS,side) = kron(boundary(mps1,side), boundary(mps2,side))

transfer_matrix_bond(mps::AbstractOpenMPS, site::Integer,dir::Symbol) = I
transfer_matrix_bond(mps1::AbstractOpenMPS,mps2::AbstractOpenMPS, site::Integer,dir::Symbol) = I
transfer_matrix_bond(mps::ConjugateMPS{<:AbstractOpenMPS}, site::Integer,dir::Symbol) = I
transfer_matrix_bond(mps1::ConjugateMPS{<:AbstractOpenMPS},mps2::AbstractOpenMPS, site::Integer,dir::Symbol) = I

"""
    expectation_values(mps::AbstractOpenMPS, op)

Return a list of expectation values on every site

See also: [`expectation_value`](@ref)
"""
function expectation_values(mps::Union{AbstractOpenMPS,MPSSum}, op; string=IdentityMPOsite)
    opLength = length(op)
    N = length(mps)
    vals = Array{ComplexF64,1}(undef, N + 1 - opLength)
    for site = 1:N+1-opLength
        vals[site] = expectation_value(mps,op,site, string=string)
    end
    return vals
end
function expectation_values(mps::AbstractOpenMPS, op::Vector{T}; string=IdentityMPOsite) where {T}
    opLength = length(op)
    N = length(mps)
    @assert N == opLength + length(op[end])-1
    vals = Array{ComplexF64,1}(undef,N-length(op[end])+1)
    for site = 1:N-length(op[end])+1
        vals[site] = expectation_value(mps,op[site],site; string=string)
    end
    return vals
end


"""
    correlator(mps,op1,op2)

Return the two-site expectation values

See also: [`connected_correlator`](@ref)
"""
function correlator(mps::AbstractOpenMPS, op1, op2, k1::Integer, k2::Integer)
    N = length(mps.Γ)
	oplength1 = length(op1)
	oplength2 = length(op2)

	emptytransfers = transfer_matrices(mps,:left)
	op1transfers = map(site -> transfer_matrix(mps,op1,site,:left),1:N-oplength1+1)
    op2transfers = map(site -> transfer_matrix(mps,op2,site,:left),1:N-oplength2+1)

    function idR(n)
        d = size(mps.Γ[n],3)
        return vec(Matrix(1.0I,d,d))
    end

    corr = zeros(ComplexF64,N-oplength1+1,N-oplength2+1)
    for n2 in k2:-1:oplength1+1
        L = op2transfers[n2]*idR(n2+oplength2-1)
        for n1 in n2-oplength1:-1:k1
            Λ2 = mps.Λ[n1].^2
            L2 = reshape(op1transfers[n1]*L,length(Λ2),length(Λ2))
            corr[n1,n2] = tr(Diagonal(Λ2)*L2)
            L = emptytransfers[n1+oplength1-1]*L
        end
    end
    for n2 in k2:-1:oplength2+1
        L = op1transfers[n2]*idR(n2+oplength1-1)
        for n1 in n2-oplength2:-1:k1
            Λ2 = mps.Λ[n1].^2
            L2 = reshape(op2transfers[n1]*L,length(Λ2),length(Λ2))
            corr[n2,n1] = tr(Diagonal(Λ2)*L2)
            L = emptytransfers[n1+oplength2-1]*L
        end
    end
	return corr[k1:k2,k1:k2]
end
function correlator(mps::AbstractOpenMPS,op1,op2) 
    N = length(mps.Γ)
    oplength1 = length(op1)
	oplength2 = length(op2)
    correlator(mps::OpenMPS,op1,op2,1,N+1-max(oplength1,oplength2))
end
function correlator(mps::AbstractOpenMPS,op)
    N = length(mps.Γ)
    oplength = length(op)
    correlator(mps::OpenMPS,op,1,N+1-oplength)
end
function correlator(mps::AbstractOpenMPS, op, k1::Integer, k2::Integer)
    N = length(mps.Γ)
	oplength = length(op)
	emptytransfers = transfer_matrices(mps,:left)
	optransfers = map(site -> transfer_matrix(mps,op,site,:left),1:N-oplength+1)
    function idR(n)
        d = size(mps.Γ[n],3)
        return vec(Matrix(1.0I,d,d))
    end
    corr = zeros(ComplexF64,N-oplength+1,N-oplength+1)
    for n2 in k2:-1:oplength+1
        L = optransfers[n2]*idR(n2+oplength-1)
        for n1 in n2-oplength:-1:k1
            Λ2 = mps.Λ[n1].^2
            L2 = reshape(optransfers[n1]*L,length(Λ2),length(Λ2))
            corr[n1,n2] = tr(Diagonal(Λ2)*L2)
            L = emptytransfers[n1+oplength-1]*L
            corr[n2,n1] = corr[n1,n2]
        end
    end
	return corr[k1:k2,k1:k2]
end

function connected_correlator(mps::AbstractOpenMPS,op1,op2)
    N = length(mps.Γ)
    oplength1 = length(op1)
	oplength2 = length(op2)
    return connected_correlator(mps,op1,op2,1,N+1-max(oplength1,oplength2))
end
function connected_correlator(mps::AbstractOpenMPS,op)
    N = length(mps.Γ)
    oplength = length(op)
    return connected_correlator(mps,op,1,N+1-oplength)
end
function connected_correlator(mps::AbstractOpenMPS, op1, op2, k1::Integer, k2::Integer)
    corr = correlator(mps,op1,op2,k1,k2)
    N = length(mps.Γ)
    oplength1 = length(op1)
	oplength2 = length(op2)
    ev1 = map(k->expectation_value(mps,op1,k), k1:k2)
    ev2 = map(k->expectation_value(mps,op2,k), k1:k2)
    concorr=zeros(ComplexF64,k2-k1+1,k2-k1+1)
    for n1 in 1:(k2-k1+1-oplength2)
        for n2 in n1+oplength1:(k2-k1+1)
            concorr[n1,n2] = corr[n1,n2] - ev1[n1]*ev2[n2]
        end
    end
    for n1 in 1:(k2-k1+1-oplength1)
        for n2 in n1+oplength2:(k2-k1+1)
            concorr[n2,n1] = corr[n2,n1] - ev2[n1]*ev1[n2]
        end
    end
    return concorr
end
function connected_correlator(mps::AbstractOpenMPS, op, k1::Integer, k2::Integer)
    corr = correlator(mps,op,k1,k2)
    N = length(mps.Γ)
    oplength = length(op)
    ev = pmap(k->expectation_value(mps,op,k), k1:k2)
    concorr=zeros(ComplexF64,k2-k1+1,k2-k1+1)
    for n1 in 1:(k2-k1+1-oplength)
        for n2 in n1+oplength:(k2-k1+1)
            concorr[n1,n2] = corr[n1,n2] - ev[n1]*ev[n2]
            concorr[n2,n1] = concorr[n1,n2]
        end
    end
    return concorr
end
