
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
        Tc = transfer_matrix_bond(mps,site, :left)
        T = transfer_matrix(mps[site:site+n-1], op, :left)
        return transpose(Tc*( T* R)) * L
    else 
        return expectation_value(mps[site:site + n - 1], op)
    end
end

function expectation_value(mps::AbstractMPS, mpo::AbstractMPO) 
    @assert length(mps) == length(mpo)
    L = vec(boundary(mps,mpo,:left) )
    R = vec(boundary(mps,mpo,:right))
    T = transfer_matrix(mps,mpo,:left)
    Tc = transfer_matrix_bond(mps, 1, :left)
    return (Tc*(T*R))'*L
end

function matrix_element(mps1::AbstractMPS, op, mps2::AbstractMPS, site::Integer; string=IdentityMPOsite)
    n = length(op)
    L = boundary(mps1, mps2, :left)
    R = boundary(mps1, mps2, :right)
    for k in 1:site - 1
        L = transfer_matrix(mps1[k], string, mps2[k], :right) * L
    end
    for k in length(mps1):-1:site + n
        R = transfer_matrix(mps1[k], mps2[k], :left) * R
    end
    T = transfer_matrix(mps1[site:site+n-1], op,mps2[site:site+n-1], :left)
    Tc = transfer_matrix_bond(mps1, mps2, site, :left)
    return transpose(Tc*(T* R))*L
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
    Λ = Diagonal(data(sites[1].Λ1).^2)
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
