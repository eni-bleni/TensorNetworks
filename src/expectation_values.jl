
"""
expectation_value(mps::AbstractOpenMPS, op::AbstractGate, site::Integer; iscanonical=true, string=IdentityGate(1))

Return the expectation value of the gate starting at `site`
"""
function expectation_value(mps::AbstractMPS, op, site::Integer; iscanonical=false, string=IdentityMPOsite) 
    n = length(op)
    if !iscanonical || string != IdentityMPOsite
        L = Array(vec(boundary(mps,:left)))
        R = Array(vec(boundary(mps,:right)))
        for k in 1:site - 1
            L = transfer_matrix(mps[k], string, :right) * L
        end
        for k in length(mps):-1:site + n
            R = transfer_matrix(mps[k], :left) * R
        end
        Tc = transfer_matrix_bond(mps',mps,site, :left)
        T = transfer_matrix(mps[site:site+n-1], op, :left)
        return transpose(Tc*( T* R)) * L
    else 
        return expectation_value(mps[site:site + n - 1], op)
    end
end

function expectation_value(mps::AbstractMPS, mpo::AbstractMPO) 
    @assert length(mps) == length(mpo) "Length of mps is not equal to length of mpo"
    K = numtype(mps)
    L = vec(boundary(mps,mpo,:left))
    R::Vector{K} = vec(boundary(mps,mpo,:right))
    Ts::Vector{LinearMap{K}} = transfer_matrices(mps,mpo,:left)
    Tc = transfer_matrix_bond(mps, 1, :right)
    for k in length(mps):-1:1
        R = Ts[k]*R
    end
    return transpose(R)*(Tc*L)
end

function matrix_element(mps1::BraOrKet, op, mps2::BraOrKet, site::Integer; string=IdentityMPOsite)
    n = length(op)
    K = numtype(mps1,mps2)
    L::Vector{K} = boundary(mps1, mps2, :left)
    R::Vector{K} = boundary(mps1, mps2, :right)
    for k in 1:site - 1
        L = transfer_matrix(mps1[k], string, mps2[k], :right) * L
    end
    for k in length(mps1):-1:site + n
        R = transfer_matrix(mps1[k], mps2[k], :left) * R
    end
    T::LinearMap{K} = transfer_matrix(mps1[site:site+n-1], op,mps2[site:site+n-1], :left)
    Tc = transfer_matrix_bond(mps1, mps2, site, :left)
    return (transpose(Tc*(T* R))*L)::K
end

function expectation_value(mps::MPSSum, op, site::Integer; string=IdentityMPOsite)
    #FIXME define matrix_element. Decide if "site" argument should be included or not. Decide on gate or mpo
    #Define alias Operator as Union{(MPOsite, site), MPO, Gate, Gates}?
    
    states = mps.states
    N = length(states)
    res = zero(eltype(data(states[1][2][1])))
    isherm = ishermitian(op) && ishermitian(string) #Save some computational time?
    for n in 1:N
        for k in n:N
            m = conj(states[n][1])*states[k][1]*matrix_element(states[n][2]', op, states[k][2], site; string=string)
            if k==n
                res += m
            elseif isherm
                res += 2*real(m)
            else
                res += m + conj(states[k][1])*states[n][1]*matrix_element(states[k][2]', op, states[n][2], site; string=string)
            end
        end
    end
    return res
end

# expectation_value(sites::AbstractMPS, ::IdentityGate) = expectation_value(mps[site:site+opLength-1], Ide)

# function expectation_value(sites::Vector{GenericSite}, gate::AbstractSquareGate{T,N}) where {T,N}
#     @assert length(sites) == N "Error in 'expectation value': length(sites) != length(gate)"
#     transfer_matrix(sites,gate,:left)
# end


function expectation_value(sites::Vector{OrthogonalLinkSite{T}}, gate::AbstractSquareGate) where {T}
    @assert length(sites) == length(gate)
    Λ = data(sites[1].Λ1).^2
    transfer = transfer_matrix(sites, gate, :left)
    DR = size(sites[end], 3)
    idR = vec(Matrix{T}(I, DR, DR))
    return vec(Λ)' * (transfer * idR)
end
function expectation_value(sites::Vector{GenericSite{T}}, gate::AbstractSquareGate) where {T}
    @assert length(sites) == length(gate) "Error in 'expectation value': length(sites) != length(gate)"
    transfer = transfer_matrix(sites, gate, :left)
    DL = size(sites[1], 1)
    DR = size(sites[end], 3)
    idL = vec(Matrix{T}(I, DL, DL))'
    idR = vec(Matrix{T}(I, DR, DR))
    return idL * (transfer * idR)
end


"""
    expectation_values(mps::AbstractMPS, op)

Return a list of expectation values on every site

See also: [`expectation_value`](@ref)
"""
function expectation_values(mps::Union{AbstractMPS,MPSSum}, op; string=IdentityMPOsite)
    opLength = length(op)
    N = length(mps)
    return [expectation_value(mps,op,site, string=string) for site in 1:N+1-opLength]
end

function expectation_values(mps::AbstractMPS, op::Vector{T}; string=IdentityMPOsite) where {T}
    opLength = length(op)
    N = length(mps)
    @assert N == opLength + length(op[end])-1
    return [expectation_value(mps,op[site],site; string=string) for site in 1:N-length(op[end])+1]
end

"""
    correlator(mps,op1,op2)

Return the two-site expectation values

See also: [`connected_correlator`](@ref)
"""
function correlator(mps::AbstractMPS, op1, op2, k1::Integer, k2::Integer; string = IdentityGate(1))
    N = length(mps)
	oplength1 = length(op1)
	oplength2 = length(op2)

	emptytransfers = transfer_matrices(mps,string,:left)
	op1transfers = transfer_matrices(mps,op1,:left)[1:N-oplength1+1]#map(site -> transfer_matrix(mps,op1,site,:left),1:N-oplength1+1)
    op2transfers = transfer_matrices(mps,op2,:left)[1:N-oplength2+1]#map(site -> transfer_matrix(mps,op2,site,:left),1:N-oplength2+1) 
    op1stringtransfers = transfer_matrices(mps,op1*repeatedgate(string,oplength1),:left)[1:N-oplength1+1]
    op2stringtransfers = transfer_matrices(mps,op2*repeatedgate(string,oplength2),:left)[1:N-oplength2+1]
    function idR(n)
        d = size(mps.Γ[n],3)
        return vec(Matrix{eltype(mps.Γ[n])}(I,d,d))
    end
    function idL(n)
        d = size(mps.Γ[n],1)
        return vec(Matrix{eltype(mps.Γ[n])}(I,d,d))
    end

    corr = zeros(eltype(mps[1]),N-oplength1+1,N-oplength2+1)
    for n2 in k2:-1:oplength1+1 #Op2 is on the right
        L = op2transfers[n2]*idR(n2+oplength2-1)
        for n1 in n2-oplength1:-1:k1
            Λ2 = transfer_matrix_bond(mps,n1,:left)  # = mps.Λ[n1].^2
            # L2 = reshape(op1transfers[n1]*L,length(Λ2),length(Λ2))
            # corr[n1,n2] = tr(Λ2*L2)
            L2 = op1stringtransfers[n1]*L #String operator intersects with the left operator
            corr[n1,n2] = idL(n1)'*Λ2*L2
            L = emptytransfers[n1+oplength1-1]*L
        end
    end
    for n2 in k2:-1:oplength2+1 #Op1 is on the right
        L = op1transfers[n2]*idR(n2+oplength1-1)
        for n1 in n2-oplength2:-1:k1
            Λ2 = transfer_matrix_bond(mps,n1,:left)#mps.Λ[n1].^2
            # L2 = reshape(op2transfers[n1]*L,length(Λ2),length(Λ2))
            # corr[n2,n1] = tr(Λ2*L2)
            L2 = op2stringtransfers[n1]*L
            corr[n2,n1] = idL(n1)'*Λ2*L2
            L = emptytransfers[n1+oplength2-1]*L
        end
    end
	return corr[k1:k2,k1:k2]
end
function correlator(mps::AbstractMPS,op1,op2) 
    N = length(mps.Γ)
    oplength1 = length(op1)
	oplength2 = length(op2)
    correlator(mps,op1,op2,1,N+1-max(oplength1,oplength2))
end
function correlator(mps::AbstractMPS,op)
    N = length(mps.Γ)
    oplength = length(op)
    correlator(mps,op,1,N+1-oplength)
end
function correlator(mps::AbstractMPS, op, k1::Integer, k2::Integer)
    N = length(mps.Γ)
	oplength = length(op)
	emptytransfers = transfer_matrices(mps,:left)
	optransfers = map(site -> transfer_matrix(mps,op,site,:left),1:N-oplength+1)
    function idR(n)
        d = size(mps.Γ[n],3)
        return vec(Matrix(1.0I,d,d))
    end
    corr = zeros(eltype(mps[1]),N-oplength+1,N-oplength+1)
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

function connected_correlator(mps::AbstractMPS,op1,op2)
    N = length(mps.Γ)
    oplength1 = length(op1)
	oplength2 = length(op2)
    return connected_correlator(mps,op1,op2,1,N+1-max(oplength1,oplength2))
end
function connected_correlator(mps::AbstractMPS,op)
    N = length(mps.Γ)
    oplength = length(op)
    return connected_correlator(mps,op,1,N+1-oplength)
end
function connected_correlator(mps::AbstractMPS, op1, op2, k1::Integer, k2::Integer)
    corr = correlator(mps,op1,op2,k1,k2)
    N = length(mps.Γ)
    oplength1 = length(op1)
	oplength2 = length(op2)
    ev1 = map(k->expectation_value(mps,op1,k), k1:k2)
    ev2 = map(k->expectation_value(mps,op2,k), k1:k2)
    concorr=zeros(eltype(mps[1]),k2-k1+1,k2-k1+1)
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
function connected_correlator(mps::AbstractMPS, op, k1::Integer, k2::Integer)
    corr = correlator(mps,op,k1,k2)
    N = length(mps.Γ)
    oplength = length(op)
    ev = pmap(k->expectation_value(mps,op,k), k1:k2)
    concorr=zeros(eltype(mps[1]),k2-k1+1,k2-k1+1)
    for n1 in 1:(k2-k1+1-oplength)
        for n2 in n1+oplength:(k2-k1+1)
            concorr[n1,n2] = corr[n1,n2] - ev[n1]*ev[n2]
            concorr[n2,n1] = concorr[n1,n2]
        end
    end
    return concorr
end
