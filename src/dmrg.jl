"""
    DMRG(mpo, mps_input, orth=[], prec)

Use DMRG to calculate the lowest energy eigenstate orthogonal to `orth`
"""
function DMRG(mpo::AbstractMPO, mps_input::LCROpenMPS{T}, orth::Vector{LCROpenMPS{T}}=LCROpenMPS{T}[]; alpha=0.0, precision=DEFAULT_DMRG_precision) where {T}
    ### input: canonical random mps
    ### output: ground state mps, ground state energy
    mps = deepcopy(mps_input)
    #canonicalize!(mps)
    L = length(mps_input)
    @assert ((norm(mps_input) ≈ 1) && L == length(mpo)) "ERROR in DMRG: non-normalized MPS as input or wrong length"
    set_center!(mps,1)
    direction = :right
    Henv = environment(mps,mpo)
    orthenv = [environment(state',mps) for state in orth]
    Hsquared = multiplyMPOs(mpo,mpo)
    E, H2 = real(expectation_value(mps, mpo)), real(expectation_value(mps, Hsquared))
    var = H2 - E^2
    println("E, var = ", E, ", ", var)
    count=1
    
    while count<50 #TODO max maxcount choosable
        Eprev = E
        mps, alpha = sweep(mps,mpo,Henv,orthenv,precision,direction,orth, alpha)
        direction = reverse_direction(direction)
        E, H2 = real(expectation_value(mps,mpo)), real(expectation_value(mps, Hsquared))
        #E, H2 = mpoExpectation(mps,mpo), mpoSquaredExpectation(mps,mpo)
        if isapprox(E,real(E); atol = precision)  &&  isapprox(H2,real(H2); atol=precision)
            E, H2 = real(E), real(H2)
        else
            println("ERROR: no real energies")
            return 0
        end
        var = H2 - E^2
        println("E, var = ", E, ", ", var)
        count=count+1
        if abs((Eprev-E)/E) < precision || var/E^2 < precision
            break
        end
    end

    return mps, E
end


function effective_hamiltonian(mposite, hl,hr, orthTensors)
    szmps = (size(hl,3),size(mposite,3),size(hr,3))
    function f(v) 
        A = reshape(v, szmps)
        HA = local_mul(hl,hr,mposite,A)
        overlap(o) = 100*conj(o)*(transpose(vec(o))*v)
        OA = sum(overlap, orthTensors; init = zero(A))
        return vec(HA + OA)
    end
    return LinearMap{ComplexF64}(f, prod(szmps), ishermitian=true)
end

function eigensite(site::GenericSite,mposite, hl,hr, orthTensors,prec)
    szmps=size(site)
    heff = effective_hamiltonian(mposite, hl,hr, orthTensors)
    if size(heff)[1] < 20
        evals, evecs = eigen(Matrix(heff))
        e::ComplexF64 = evals[1]
        vecmin::Vector{ComplexF64} = evecs[:,1]
    else
        evals, evecs2 = eigsolve(heff, vec(data(site)), 2, :SR, tol=prec, ishermitian=true)
        vecmin = evecs2[1]
        e = evals[1]
    end

    if !(e ≈ real(e))
        error("ERROR: complex eigenvalues")
    end
    #evals = real(evals)
    #eval_min::Float64, ind_min::Int = findmin(evals)
    #evec_min = evecs[:,ind_min]
    return GenericSite(reshape(vecmin,szmps)/norm(vecmin), site.purification), real(e)
end


""" sweeps from left to right in the DMRG algorithm """
function sweep(mps::LCROpenMPS{T}, mpo, Henv, orthenv, prec, dir, orth::Vector{LCROpenMPS{T}}=LCROpenMPS{T}[], alpha=0.0) where {T<:Number}
    L::Int = length(mps)
    N_orth = length(orth)
    eold=0.0 #TODO this was chosen arbitrarily
    if dir==:right 
        itr = 1:L-1
        dirval=1
    elseif dir==:left
        itr = L:-1:2
        dirval=-1
    else
        @error "In sweep: choose dir :left or :right"
    end
    for j in itr
    
        @assert (center(mps) == j) "The optimization step is not performed at the center of the mps: $(center(mps)) vs $j"
        orthtensors = [local_mul(transpose(orthenv[k].L[j]), transpose(orthenv[k].R[j]),orth[k][j]') for k in 1:N_orth]
        enew = transpose(transfer_matrix(mps[j]', mpo[j], mps[j]) * vec(Henv.R[j])) * vec(Henv.L[j])
        mps[j], e2 = eigensite(mps[j],mpo[j], Henv.L[j],Henv.R[j], orthtensors, prec)
        
        if alpha > 0.0
            if abs((enew-e2)/(eold-e2)) >.3 
                alpha *= .8
            else
                alpha *= 1.2
            end
            A,B = subspace_expand(alpha,mps[j],mps[j+dirval],Henv[j, reverse_direction(dir)], mpo[j],mps.truncation, dir)
            mps.center+=dirval
            mps.Γ[j] = A
            mps.Γ[j+dirval] = B
        elseif dir==:right 
            shift_center_right!(mps)
        elseif dir==:left
            shift_center_left!(mps)
        end
        eold = enew
        update! = dir==:right ? update_left_environment! : update_right_environment!
        update!(Henv,j,mps[j]',mpo[j],mps[j])
        for k in 1:N_orth
            update!(orthenv[k],j, orth[k][j]', mps[j])
        end
    end
    return mps, alpha
end

"""
    eigenstates(hamiltonian, mps, n, prec)

Return the `n` eigenstates and energies with the lowest energy
"""
function eigenstates(hamiltonian::MPO, mps::LCROpenMPS, n::Integer; precision = DEFAULT_DMRG_precision, alpha=0.0)
    T = eltype(data(mps[1]))
    states = LCROpenMPS{T}[]
    energies = Float64[]
    for k = 1:n
        @time state, E = DMRG(hamiltonian, mps, states, precision = precision, alpha=alpha)
        append!(states,[state])
        append!(energies,E)
    end
    return states, energies
end
eigenstates(hamiltonian, mps::OpenMPS, n::Integer; precision=DEFAULT_DMRG_precision, alpha=0.0) = eigenstates(hamiltonian,LCROpenMPS(mps), n, precision=precision, alpha=alpha)
DMRG(mpo::AbstractMPO, mps::OpenMPS, precision = DEFAULT_DMRG_precision, alpha=0.0) = DMRG(mpo,LCROpenMPS(mps), precision = precision, alpha=alpha)
const DEFAULT_DMRG_precision=1e-12

function expansion_term(alpha, site, env, mposite)
    @tensor P[:] := site[1,2,-3]*env[-1,4,1]*mposite[4,-2,2,-4]
    return alpha*reshape(P,size(env,1),size(site,2),size(mposite,4)*size(site,3))
end


function subspace_expand(alpha,site,nextsite,env,mposite, trunc, dir)
    if dir==:left
        site = reverse_direction(site)
        nextsite = reverse_direction(nextsite)
        mposite = reverse_direction(mposite)
    end
    ss= size(site)
    ss2 = size(nextsite)
    d = ss[2]
    P = expansion_term(alpha, data(site), env, data(mposite))
    sp = size(P)
    P0 = zeros(ComplexF64, sp[3], d, ss2[3])
    M = Array{ComplexF64,3}(undef,(ss[1],ss[2],ss[3]+sp[3]))
    B = Array{ComplexF64,3}(undef,(ss2[1]+sp[3],ss2[2],ss2[3]))
    for k in 1:d
        M[:,k,:] = hcat(data(site)[:,k,:],P[:,k,:])
        B[:,k,:] = vcat(data(nextsite)[:,k,:],P0[:,k,:])
    end
    U,S,V,err = split_truncate!(reshape(M,ss[1]*ss[2],ss[3]+sp[3]), trunc)
    newsite = GenericSite(reshape(Matrix(U),ss[1],ss[2],length(S)), false)
    newnextsite = VirtualSite(Diagonal(S)*V)*GenericSite(B,false)
    if dir==:left
        newsite = reverse_direction(newsite)
        newnextsite = reverse_direction(newnextsite)
    end
    return newsite, newnextsite
end

