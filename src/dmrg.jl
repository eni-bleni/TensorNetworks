"""
    DMRG(mpo, mps_input, orth=[], prec)

Use DMRG to calculate the lowest energy eigenstate orthogonal to `orth`
"""
function DMRG(mpo::AbstractMPO, mps_input::LCROpenMPS{T}, orth::Vector{LCROpenMPS{T}}=LCROpenMPS{T}[];kwargs...) where {T}
    ### input: canonical random mps
    ### output: ground state mps, ground state energy
    precision::Float64=get(kwargs,:precision, DEFAULT_DMRG_precision)
    mps::LCROpenMPS{T} = canonicalize(deepcopy(mps_input))
    set_center!(mps,1)
    #canonicalize!(mps)
    L = length(mps_input)
    @assert (norm(mps_input) ≈ 1 && L == length(mpo)) "ERROR in DMRG: non-normalized MPS as input or wrong length"
    direction = :right
    Henv = environment(mps,mpo)
    orthenv = [environment(state',mps) for state in orth]
    Hsquared = multiplyMPOs(mpo,mpo)
    E::real(T), H2::real(T) = real(expectation_value(mps, mpo)), real(expectation_value(mps, Hsquared))
    var = H2 - E^2
    println("E, var = ", E, ", ", var)
    count=1
   
    while count<50 #TODO make maxcount choosable
        Eprev = E
        mps = sweep(mps,mpo,Henv,orthenv,direction,orth; kwargs...)
        mps = canonicalize(mps,center=mps.center)
        direction = reverse_direction(direction)
        E, H2 = real(expectation_value(mps,mpo)), real(expectation_value(mps,Hsquared))
        #E, H2 = mpoExpectation(mps,mpo), mpoSquaredExpectation(mps,mpo)
        if isapprox(E,real(E); atol = precision)  &&  isapprox(H2,real(H2); atol=precision)
            E, H2 = real(E), real(H2)
        else
            @warn "Energies are not real"
        end
        var = H2 - E^2
        println("E, var, ΔE/E = ", E, ", ", var, ", ", (Eprev-E)/E)
        count=count+1
        if abs((Eprev-E)/E) < precision && var/E^2 < precision #&& count>10
           break
        end
    end

    return mps, E
end


function effective_hamiltonian(mposite, hl,hr, orthvecs)
    szmps = (size(hl,3),size(mposite,3),size(hr,3))
    function f(v) 
        A = reshape(v, szmps)
        HA = local_mul(hl,hr,mposite,A)
        overlap(o) = 100*o*(o'*v)
        OA = sum(overlap, orthvecs; init = zero(v))
        return vec(HA) + OA
    end
    return LinearMap{eltype(hl)}(f, prod(szmps), ishermitian=true)
end

const BigNumber = Union{ComplexDF64,ComplexDF32,ComplexDF16,Double64,Double32,Double16,BigFloat,Complex{BigFloat}}
function eigs(heff::LinearMap, x0, nev, prec)
    if prod(size(heff)) < 100
        evals, evecs = _eigs_small(Matrix(heff))
    else
        evals,evecs = _eigs_large(heff, x0, nev, prec)
    end
    return evals, evecs
end
_eigs_small(heff) = eigen(heff)
function _eigs_large(heff::LinearMap, x0, nev, prec)
    evals::Vector{Float64}, evecs::Vector{Vector{eltype(LinearMap)}} = eigsolve(heff, vec(data(x0)), nev, :SR, tol=prec, ishermitian=true, maxiter=1000)
    evecsvec::Array{eltype(LinearMap),2} = hcat(evecs...)
    return evals, evecsvec
end
_eigs_large(heff::LinearMap{<:BigNumber}, x0, nev, prec) = partialeigen(partialschur(heff,nev=nev, which =SR(), tol=prec)[1])
function eigensite(site::GenericSite, mposite, hl, hr, orthvecs, prec)
    szmps=size(site)
    heff = effective_hamiltonian(mposite, hl,hr, orthvecs)
    evals, evecs = eigs(heff,site,1,prec)
    e::eltype(hl) = evals[1]
    vecmin::Vector{eltype(hl)} = evecs[:,1]
    if !(e ≈ real(e))
        error("ERROR: complex eigenvalues")
    end
    #evals = real(evals)
    #eval_min::Float64, ind_min::Int = findmin(evals)
    #evec_min = evecs[:,ind_min]
    # println(size(evecs))
    # println(size(heff))
    # println(size(vec(data(site))))
    # println(size(site))
    # println(size(vecmin))
    return GenericSite(reshape(vecmin,szmps)/norm(vecmin), site.purification), real(e)
end

""" sweeps from left to right in the DMRG algorithm """
function sweep(mps::LCROpenMPS{T}, mpo::AbstractMPO, Henv::AbstractFiniteEnvironment, orthenv, dir, orth::Vector{LCROpenMPS{T}}=LCROpenMPS{T}[]; kwargs...) where {T}
    L::Int = length(mps)
    N_orth = length(orth)
    shifter = get(kwargs, :shifter, ShiftCenter())
    precision=get(kwargs,:precision, DEFAULT_DMRG_precision) 
    # eold=0.0 #TODO this was chosen arbitrarily
    if dir==:right 
        itr = 1:L-1
        # dirval=1
    elseif dir==:left
        itr = L:-1:2
        # dirval=-1
    else
        @error "In sweep: choose dir :left or :right"
    end
    for j in itr
        @assert (center(mps) == j) "The optimization step is not performed at the center of the mps: $(center(mps)) vs $j"
        orthvecs = [vec(data(local_mul(orthenv[k].L[j]', orthenv[k].R[j]',orth[k][j]))) for k in 1:N_orth]
        # enew = transpose(transfer_matrix(mps[j]', mpo[j], mps[j]) * vec(Henv.R[j])) * vec(Henv.L[j])
        mps[j], e2 = eigensite(mps[j],mpo[j], Henv.L[j],Henv.R[j], orthvecs, precision)
        
        shift_center!(mps,j,dir,shifter; mpo = mpo, env=Henv)
        # if alpha > 0.0
        #     if abs((enew-e2)/(eold-e2)) >.3 
        #         alpha *= .8
        #     else
        #         alpha *= 1.2
        #     end
        #     A,B = subspace_expand(alpha,mps[j],mps[j+dirval],Henv[j, reverse_direction(dir)], mpo[j],mps.truncation, dir)
        #     mps.center+=dirval
        #     mps.Γ[j] = A
        #     mps.Γ[j+dirval] = B
        # elseif dir==:right 
        #     shift_center_right!(mps)
        # elseif dir==:left
        #     shift_center_left!(mps)
        # end
        # eold = enew
        update! = dir==:right ? update_left_environment! : update_right_environment!
        update!(Henv,j,mps[j]',mpo[j],mps[j])
        for k in 1:N_orth
            update!(orthenv[k],j, orth[k][j]', mps[j])
        end
    end
    return mps#, alpha
end

"""
    eigenstates(hamiltonian, mps, n, prec)

Return the `n` eigenstates and energies with the lowest energy
"""
function eigenstates(hamiltonian::MPO, mps::LCROpenMPS{T}, n::Integer; kwargs...) where {T}
    #T = eltype(data(mps[1]))
    states = Vector{LCROpenMPS{T}}(undef,n)
    shifter0 = deepcopy(get(kwargs, :shifter, ShiftCenter()))
    energies = Vector{real(promote_type(T,eltype(hamiltonian[1])))}(undef,n)
    for k = 1:n
        @time state, E = DMRG(hamiltonian, mps, states[1:k-1]; shifter = shifter0, kwargs...)
        states[k] = state
        energies[k] = E
    end
    return states, energies
end
eigenstates(hamiltonian, mps::OpenMPS, n::Integer; kwargs...) = eigenstates(hamiltonian,LCROpenMPS(mps), n; kwargs...)
DMRG(mpo::AbstractMPO, mps::OpenMPS; kwargs...) = DMRG(mpo,LCROpenMPS(mps) ; kwargs...)
const DEFAULT_DMRG_precision=1e-12

function expansion_term(alpha, site, env, mposite)
    @tensor P[:] := data(site)[1,2,-3]*env[-1,4,1]*data(mposite)[4,-2,2,-4]
    return alpha*reshape(P,size(env,1),size(site,2),size(mposite,4)*size(site,3))
end
function expansion_term(alpha, site, env, mposite::ScaledIdentityMPOsite)
    s = size(site)
    newsite = reshape(env*reshape(data(site), s[1],s[2]*s[3]), s[1],s[2],s[3])
    return rmul!(newsite, alpha*data(mposite))
    #@tensor P[:] := data(site)[1,-2,-3]*env[-1,1]
    #return data(mposite)*alpha*reshape(P, size(env,1), size(site,2), size(site,3))
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
    P = expansion_term(alpha, site, env, mposite)
    sp = size(P)
    P0 = zeros(eltype(env), sp[3], d, ss2[3])
    M = Array{eltype(env),3}(undef,(ss[1],ss[2],ss[3]+sp[3]))
    B = Array{eltype(env),3}(undef,(ss2[1]+sp[3],ss2[2],ss2[3]))
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

