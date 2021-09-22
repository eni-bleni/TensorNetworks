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
    Lorth = length(orth) 
    @assert ((norm(mps_input) ≈ 1) && L == length(mpo)) "ERROR in DMRG: non-normalized MPS as input or wrong length"
    set_center!(mps,1)
    direction = :right
    HL, HR = initializeHLR(mps,mpo)
    CL, CR = initializeCLR(mps,orth)
    Hsquared = multiplyMPOs(mpo,mpo)
    E, H2 = real(expectation_value(mps, mpo)), real(expectation_value(mps, Hsquared))
    var = H2 - E^2
    println("E, var = ", E, ", ", var)
    count=1
    
    while count<50 #TODO max maxcount choosable
        Eprev = E
        mps, alpha = sweep(mps,mpo,HL,HR,CL,CR,precision,direction,orth,alpha)
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

function HeffFun(Avec::Vector{T},mposite,hl,hr, orthTensors::Vector{Array{T,3}}) where {T}
    szmps = (size(hl,3),size(mposite,3),size(hr,3))
    A = reshape(Avec, szmps)
    HA = HeffMult(A ,data(mposite),hl,hr)
    
    overlap(o) = 100*conj(o)*(transpose(vec(o))*Avec)
    OA = sum(overlap, orthTensors; init = zero(A))
    # for k = 1:length(orthTensors)
    #     #@tensor overlap[:] := orthTensors[k][1,2,3]*A[1,2,3]
    #     overlap = transpose(vec(orthTensors[k]))*Avec
    #     Aout += 100*conj(orthTensors[k]) * overlap #TODO make the weight choosable
    # end
    #return vec(Aout)
    return vec(HA + OA)
end
function effective_hamiltonian(mposite, hl, hr, orthTensors)
    f(v) = HeffFun(v,mposite,hl,hr,orthTensors)
    return LinearMap{ComplexF64}(f, prod((size(hl,3),size(mposite,3),size(hr,3))),ishermitian=true)
end

function get_state_overlaps(cl,cr,orth)
    N_orth = length(orth)
    orthTensors = Vector{Array{ComplexF64,3}}(undef,N_orth)
    for k = 1:N_orth
        @tensor orthTensors[k][:] := cl[k][1,-1]*cr[k][2,-3]*conj(orth[k][1,-2,2])
    end
    return orthTensors
end

function eigensite(site::GenericSite,mposite, hl,hr, orthTensors,prec)
    szmps=size(site)
    heff = effective_hamiltonian(mposite, hl, hr, orthTensors)
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

function expectation_value(op,site,envL,envR)
    @tensor e[:] := envL[1,2,3]*op[2,4,5,6]*conj(site[1,4,7])*site[3,5,8]*envR[7,6,8]
    return e[1]
end

""" sweeps from left to right in the DMRG algorithm """
function sweep(mps::LCROpenMPS{T}, mpo, HL, HR, CL, CR, prec, dir, orth::Vector{LCROpenMPS{T}}=LCROpenMPS{T}[], alpha=0.0) where {T<:Number}
    ### minimizes E by diagonalizing site by site in the mps from left to right: j=1-->L-1
    ### the resulting sites are left-canonicalized
    L::Int = length(mps)
    N_orth = length(orth)
    #orthTensors = Array{Array{ComplexF64,3},1}(undef,N_orth)
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
        cl = [CL[k][j] for k in 1:N_orth]
        cr = [CR[k][j] for k in 1:N_orth]
        o::Vector{Array{T,3}} = [data(orth[k][j]) for k in 1:N_orth]
        orthTensors = get_state_overlaps(cl,cr,o)
        enew = real.(expectation_value(data(mpo[j]),data(mps[j]),HL[j],HR[j]))
        #@tensor enew[:] := HL[j][1,2,3]*data(mpo[j])[2,4,5,6]*conj(data(mps[j])[1,4,7])*data(mps[j])[3,5,8]*HR[j][7,6,8]
        mps[j], e2 = eigensite(mps[j],mpo[j], HL[j],HR[j], orthTensors,prec)
        
        if alpha > 0.0
            if abs((enew-e2)/(eold-e2)) >.3 
                alpha *= .8
            else
                alpha *= 1.2
            end
            if dir==:right
                env=HL[j]
            else
                env = HR[j]
            end
            A,B = subspace_expand(alpha,mps[j],mps[j+dirval],env,mpo[j],mps.truncation, dir)
            mps.center+=dirval
            mps.Γ[j] = A
            mps.Γ[j+dirval] = B
            # if dir==:right
            #     A,B = subspace_expand_right(alpha,mps[j],mps[j+1],HL[j],mpo[j],mps.truncation)
            #     mps.center+=1
            #     mps.Γ[j] = A
            #     mps.Γ[j+1] = B
            # end
        elseif dir==:right 
            shift_center_right!(mps)
        elseif dir==:left
            shift_center_left!(mps)
        end
        eold = enew

        updateCLR(mps,CL,CR,j,dir,orth)
        updateHLR(mps,mpo,HL,HR,j,dir)
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


function initializeHLR(mps::LCROpenMPS,mpo)
    L = length(mps)
    T = eltype(data(mps[1]))
    HL = Vector{Array{T,3}}(undef,L)
    HR = Vector{Array{T,3}}(undef,L)
    HR[L] = Array{T}(undef,1,1,1)
    HR[L][1,1,1] = 1
    HL[1] = Array{T}(undef,1,1,1)
    HL[1][1,1,1] = 1
    for j=L-1:-1:1
        # @tensoropt (-1,1,-3,3) HR[j][-1,-2,-3] := conj(data(mps[j+1])[-1,4,1])*mpo[j+1].data[-2,4,5,2]*data(mps[j+1])[-3,5,3]*HR[j+1][1,2,3]
        #@tensor HR[j][-1,-2,-3] := conj(data(mps[j+1])[-1,4,1])*mpo[j+1].data[-2,4,5,2]*data(mps[j+1])[-3,5,3]*HR[j+1][1,2,3]
        HR[j] = calc_HR_env(data(mps[j+1]),data(mpo[j+1]),HR[j+1])
    end
    for j=2:L
        # @tensoropt (1,3,-1,-3) HL[j][-1,-2,-3] := HL[j-1][1,2,3]*conj(data(mps[j-1])[1,4,-1])*mpo[j-1].data[2,4,5,-2]*data(mps[j-1])[3,5,-3]
        #@tensor HL[j][-1,-2,-3] := HL[j-1][1,2,3]*conj(data(mps[j-1])[1,4,-1])*mpo[j-1].data[2,4,5,-2]*data(mps[j-1])[3,5,-3]
        HL[j] = calc_HL_env(data(mps[j-1]),data(mpo[j-1]),HL[j-1])
    end
    return HL, HR
end

function calc_HR_env(site, mposite, Hin)
    return @tensor Hout[:] := conj(site[-1,4,1])*mposite[-2,4,5,2]*site[-3,5,3]*Hin[1,2,3]
end
function calc_HL_env(site, mposite, Hin)
    return @tensor Hout[:] := conj(site[1,4,-1])*mposite[2,4,5,-2]*site[3,5,-3]*Hin[1,2,3]
end

function initializeCLR(mps::LCROpenMPS{T},orth=LCROpenMPS{T}[]) where {T}
    L = length(mps)
    Lorth = length(orth)
    #T = eltype(data(mps[1]))
    CL = Vector{Array{Array{T,2},1}}(undef,Lorth)
    CR = Vector{Array{Array{T,2},1}}(undef,Lorth)
    for k = 1:length(orth)
        CR[k] = Array{Array{T,2}}(undef,L)
        CL[k] = Array{Array{T,2}}(undef,L)
        CR[k][L] = Array{T}(undef,1,1)
        CR[k][L][1,1] = 1
        CL[k][1] = Array{T}(undef,1,1)
        CL[k][1][1,1] = 1
        mps2::LCROpenMPS{T} = orth[k]
        for j=1:L-1
            # @tensoropt (-2,2,-1,3) CR[k][L-j][-1,-2] := data(mps[L-j+1])[-2,1,2]*conj(data(mps2[L-j+1])[-1,1,3])*CR[k][L-j+1][3,2]
            # @tensoropt (2,-2,1,-1) CL[k][1+j][-1,-2] := data(mps[j])[2,3,-2]*conj(data(mps2[j])[1,3,-1])*CL[k][j][1,2]
            
            #@tensor CR[k][L-j][-1,-2] := data(mps[L-j+1])[-2,3,2]*conj(data(mps2[L-j+1])[-1,3,1])*CR[k][L-j+1][1,2]
            CR[k][L-j] = calc_CR_env(data(mps[L-j+1]),data(mps2[L-j+1]),CR[k][L-j+1])
            #@tensor CL[k][1+j][-1,-2] := data(mps[j])[2,3,-2]*conj(data(mps2[j])[1,3,-1])*CL[k][j][1,2]
            CL[k][j+1] = calc_CL_env(data(mps[j]),data(mps2[j]),CL[k][j])
        end
    end
    return CL, CR
end
function calc_CR_env(site,site2, Cin)
    return @tensor Cout[:] := site[-2,3,2]*conj(site2[-1,3,1])*Cin[1,2]
end
function calc_CL_env(site,site2, Cin)
    return @tensor Cout[:] := site[2,3,-2]*conj(site2[1,3,-1])*Cin[1,2]
end

""" Update HL, HR, when tensor i has been updated in a dir-sweep"""
function updateHLR(mps,mpo,HL,HR,i,dir)
    L = length(mps)
    if dir==:right
        # @tensoropt (1,3,-1,-3) HL[i+1][-1,-2,-3] := HL[i][1,2,3]*conj(data(mps[i])[1,4,-1])*mpo[i].data[2,4,5,-2]*data(mps[i])[3,5,-3]
        #@tensor HL[i+1][-1,-2,-3] := (HL[i][1,5,3]*conj(data(mps[i])[1,4,-1]))*(mpo[i].data[5,4,2,-2]*data(mps[i])[3,2,-3])
        HL[i+1] = calc_HL_env(data(mps[i]),data(mpo[i]),HL[i])
    elseif dir==:left
        # @tensoropt (-1,1,-3,3) HR[i-1][-1,-2,-3] := conj(data(mps[i])[-1,4,1])*mpo[i].data[-2,4,5,2]*data(mps[i])[-3,5,3]*HR[i][1,2,3]
        #@tensor HR[i-1][-1,-2,-3] := (conj(data(mps[i])[-1,2,3])*mpo[i].data[-2,2,5,4])*(data(mps[i])[-3,5,1]*HR[i][3,4,1])
        HR[i-1] = calc_HR_env(data(mps[i]),data(mpo[i]),HR[i])
    end

end

function updateCLR(mps::LCROpenMPS,CL,CR,i, dir, orth=[])
    for k = 1:length(orth)
        Γo = orth[k]
        if dir==:right
            # @tensoropt (2,-2,1,-1) CL[k][i+1][-1,-2] := data(mps[i])[2,3,-2]*conj(data(Γo[i])[1,3,-1])*CL[k][i][1,2]
            #@tensor CL[k][i+1][-1,-2] := data(mps[i])[2,3,-2]*conj(data(Γo[i])[1,3,-1])*CL[k][i][1,2]
            CL[k][i+1] = calc_CL_env(data(mps[i]),data(Γo[i]),CL[k][i])
        elseif dir==:left
            # @tensoropt (-1,-2,3,2) CR[k][i-1][-1,-2] := data(mps[i])[-2,1,2]*conj(data(Γo[i])[-1,1,3])*CR[k][i][3,2]
            #@tensor CR[k][i-1][-1,-2] := data(mps[i])[-2,3,2]*conj(data(Γo[i])[-1,3,1])*CR[k][i][1,2]
            CR[k][i-1] = calc_CR_env(data(mps[i]),data(Γo[i]),CR[k][i])
        end
    end
end

# function getHeff(mps,mpo,HL,HR,i)
#     L=length(mps)
#     @tensor Heff[:] := HL[i][-1,1,-4]*mpo[i].data[1,-2,-5,2]*HR[i][-3,2,-6]
#     return Heff
# end

function HeffMult(tensor,mposite,HL,HR)
    #@tensoropt (-1,4,6,-3) temp[:] := HL[-1,1,4]* mposite[1,-2,5,2] *tensor[4,5,6]*HR[-3,2,6]
    @tensor temp[:] := (HL[-1,2,3]* mposite[2,-2,4,5]) *(tensor[3,4,1]*HR[-3,5,1])
    return temp
end

function expansion_term(alpha, site, hl, mposite)
    @tensor P[:] := site[1,2,-3]*hl[-1,4,1]*mposite[4,-2,2,-4]
    return alpha*reshape(P,size(hl,1),size(site,2),size(mposite,4)*size(site,3))
end

# function subspace_expand_right(alpha,site,nextsite,hl,mposite, trunc)
#     ss= size(site)
#     ss2 = size(nextsite)
#     d = ss[2]
#     P = expansion_term(alpha, data(site), hl, data(mposite))
#     sp = size(P)
#     P0 = zeros(ComplexF64, sp[3], d, ss2[3])
#     M = Array{ComplexF64,3}(undef,(ss[1],ss[2],ss[3]+sp[3]))
#     B = Array{ComplexF64,3}(undef,(ss2[1]+sp[3],ss2[2],ss2[3]))
#     for k in 1:d
#         M[:,k,:] = hcat(data(site)[:,k,:],P[:,k,:])
#         B[:,k,:] = vcat(data(nextsite)[:,k,:],P0[:,k,:])
#     end
#     U,S,V,err = split_truncate!(reshape(M,ss[1]*ss[2],ss[3]+sp[3]), trunc)
#     newsite = GenericSite(reshape(Matrix(U),ss[1],ss[2],length(S)), false)
#     newnextsite = VirtualSite(Diagonal(S)*V)*GenericSite(B,false)
#     return newsite, newnextsite
# end

function subspace_expand(alpha,site,nextsite,hl,mposite, trunc, dir)
    if dir==:left
        site = reverse_direction(site)
        nextsite = reverse_direction(nextsite)
        mposite = reverse_direction(mposite)
    end
    ss= size(site)
    ss2 = size(nextsite)
    d = ss[2]
    P = expansion_term(alpha, data(site), hl, data(mposite))
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

