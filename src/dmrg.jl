"""
    DMRG(mps_input, mpo, prec, orth=[])

Use DMRG to calculate the lowest energy eigenstate orthogonal to `orth`
"""
function DMRG(mps_input::OrthOpenMPS{T}, mpo::AbstractMPO, prec, orth=OrthOpenMPS{T}[]::Vector{OrthOpenMPS{T}}) where {T}
    ### input: canonical random mps
    ### output: ground state mps, ground state energy
    mps = copy(mps_input)
    canonicalize!(mps)
    L = length(mps_input)
    Lorth = length(orth) 
    @assert ((norm(mps_input) ≈ 1) && L == length(mpo)) "ERROR in DMRG: non-normalized MPS as input or wrong length"
    if check_LRcanonical(mps[L],:right)
        canonicity=:right
    elseif check_LRcanonical(mps[1],:left)
        canonicity=:left
    else
        println("ERROR in DMRG: use canonical mps as input")
        return 0
    end

    HL = Array{Array{T,3},1}(undef,L)
    HR = Array{Array{T,3},1}(undef,L)
    CL = Array{Array{Array{T,2},1},1}(undef,Lorth)
    CR = Array{Array{Array{T,2},1},1}(undef,Lorth)
    initializeHLR(mps,mpo,HL,HR)
    initializeCLR(mps,CL,CR,orth)
    Hsquared = multiplyMPOs(mpo,mpo)
    E, H2 = real(expectation_value(mps, mpo)), real(expectation_value(mps, Hsquared))
    var = H2 - E^2
    println("E, var = ", E, ", ", var)
    count=1
    while var > prec && count<50
        Eprev = E
        mps = sweep(mps,mpo,HL,HR,CL,CR,prec,canonicity,orth)
        canonicity = switch_canonicity(canonicity)
        E, H2 = real(expectation_value(mps,mpo)), real(expectation_value(mps, Hsquared))
        #E, H2 = mpoExpectation(mps,mpo), mpoSquaredExpectation(mps,mpo)
        if isapprox(E,real(E); atol = prec)  &&  isapprox(H2,real(H2); atol=prec)
            E, H2 = real(E), real(H2)
        else
            println("ERROR: no real energies")
            return 0
        end
        var = H2 - E^2
        println("E, var = ", E, ", ", var)
        count=count+1
        if abs((Eprev-E)/E) < prec
            break
        end
    end

    return mps, E
end

function switch_canonicity(c)
    if c==:right
        return :left
    elseif c==:left
        return :right
    else
        error("Not a canonicity")
    end
end

""" sweeps from left to right in the DMRG algorithm """
function sweep(mps::OrthOpenMPS{T}, mpo, HL, HR, CL, CR, prec,canonicity, orth=OrthOpenMPS{T}[]) where {T}
    ### minimizes E by diagonalizing site by site in the mps from left to right: j=1-->L-1
    ### the resulting sites are left-canonicalized
    L = length(mps)
    N_orth = length(orth)
    orthTensors = Array{Array{ComplexF64,3},1}(undef,N_orth)
    for j = 1:L-1
        if canonicity==:left
            j=L+1-j
        end
        szmps = size(mps[j])
        mpsguess = vec(mps.Γ[j])
        #HeffFun(vec) = reshape(HeffMult(reshape(vec,szmps),mpo[j],HL[j],HR[j]),prod(szmps))

        for k = 1:N_orth
            @tensoropt (1,-1,2,-3) orthTensors[k][:] := CL[k][j][1,-1]*CR[k][j][2,-3]*conj(orth[k].Γ[j][1,-2,2])
        end
        function HeffFun(Avec)
            A = reshape(Avec, szmps)
            Aout = HeffMult(A,mpo[j],HL[j],HR[j])
            for k = 1:length(orth)
                @tensor overlap[:] := orthTensors[k][1,2,3]*A[1,2,3]
                Aout += 100*conj(orthTensors[k]) .* overlap
            end
            return vec(Aout)
        end
        hefflin = LinearMap{ComplexF64}(HeffFun, prod(szmps),ishermitian=true)

        if size(hefflin)[1] < 20
            evals, evecs = eigen(Matrix(hefflin))
        else
            #println(norm(hefflin*mpsguess))
            #evals, evecs = eigs(hefflin,nev=2,which=:SR,tol=prec,v0=mpsguess)
            evals, evecs = eigsolve(hefflin, mpsguess, 2, :SR, tol=prec, ishermitian=true)
            evecs = hcat(evecs...)
        end

        if !(evals ≈ real(evals))
            println("ERROR: no real eigenvalues")
            return 0
        end
        evals = real(evals)
        eval_min, ind_min = findmin(evals)
        evec_min = evecs[:,ind_min]

        Mj = reshape(evec_min,szmps)
        Aj,R = LRcanonical(Mj,switch_canonicity(canonicity))
        mps[j] = Aj
        if canonicity==:right
            @tensor mps.Γ[j+1][-1,-2,-3] := R[-1,1]*mps.Γ[j+1][1,-2,-3];
        elseif canonicity==:left
            @tensor mps.Γ[j-1][-1,-2,-3] := R[1,-3]*mps.Γ[j-1][-1,-2,1];
        end
        updateCLR(mps,CL,CR,j,canonicity,orth)
        updateHLR(mps,mpo,HL,HR,j,canonicity)
    end
    return mps
end

"""
    eigenstates(mps, hamiltonian, prec, n)

Return the `n` eigenstates and energies with the lowest energy
"""
function eigenstates(mps::OrthOpenMPS{T}, hamiltonian, prec,n) where {T}
    states = OrthOpenMPS{T}[]
    energies = Float64[]
    for k = 1:n
        @time state, E = DMRG(mps,hamiltonian,prec,states)
        append!(states,[state])
        append!(energies,E)
    end
    return states, energies
end
eigenstates(mps::OpenMPS, hamiltonian, prec,n) = eigenstates(OrthOpenMPS(mps), hamiltonian, prec,n)
DMRG(mps::OpenMPS, mpo::AbstractMPO, prec, orth=OrthOpenMPS{T}[]) where {T} = DMRG(OrthOpenMPS(mps),mpo, prec, orth)



function initializeHLR(mps::OrthOpenMPS{T},mpo,HL,HR) where {T}
    L = length(mps)

    HR[L] = Array{T}(undef,1,1,1)
    HR[L][1,1,1] = 1
    HL[1] = Array{T}(undef,1,1,1)
    HL[1][1,1,1] = 1
    Γ = mps.Γ
    for j=L-1:-1:1
        @tensoropt (-1,1,-3,3) HR[j][-1,-2,-3] := conj(Γ[j+1][-1,4,1])*mpo[j+1].data[-2,4,5,2]*Γ[j+1][-3,5,3]*HR[j+1][1,2,3]
    end
    for j=2:L
        @tensoropt (1,3,-1,-3) HL[j][-1,-2,-3] := HL[j-1][1,2,3]*conj(Γ[j-1][1,4,-1])*mpo[j-1].data[2,4,5,-2]*Γ[j-1][3,5,-3]
    end
end

function initializeCLR(mps::OrthOpenMPS{T},CL,CR,orth=OrthOpenMPS{T}[]) where {T}
    L = length(mps)
    Γ = mps.Γ
    for k = 1:length(orth)
        CR[k] = Array{Array{T,2}}(undef,L)
        CL[k] = Array{Array{T,2}}(undef,L)
        CR[k][L] = Array{T}(undef,1,1)
        CR[k][L][1,1] = 1
        CL[k][1] = Array{T}(undef,1,1)
        CL[k][1][1,1] = 1
        Γo = orth[k].Γ
        for j=1:L-1
            @tensoropt (-2,2,-1,3) CR[k][L-j][-1,-2] := Γ[L-j+1][-2,1,2]*conj(Γo[L-j+1][-1,1,3])*CR[k][L-j+1][3,2]
            @tensoropt (2,-2,1,-1) CL[k][1+j][-1,-2] := Γ[j][2,3,-2]*conj(Γo[j][1,3,-1])*CL[k][j][1,2]

        end
    end
end

""" Update HL, HR, when tensor i has been updated in a dir-sweep"""
function updateHLR(mps,mpo,HL,HR,i,dir)
    L = length(mps)
    Γ = mps.Γ
    if dir==:right
        @tensoropt (1,3,-1,-3) HL[i+1][-1,-2,-3] := HL[i][1,2,3]*conj(Γ[i][1,4,-1])*mpo[i].data[2,4,5,-2]*Γ[i][3,5,-3]
    end
    if dir==:left
        @tensoropt (-1,1,-3,3) HR[i-1][-1,-2,-3] := conj(Γ[i][-1,4,1])*mpo[i].data[-2,4,5,2]*Γ[i][-3,5,3]*HR[i][1,2,3]
    end

end

function updateCLR(mps::OrthOpenMPS{T},CL,CR,i,dir,orth=OrthOpenMPS{T}[]) where {T}
    L = length(mps)
    Γ = mps.Γ
    for k = 1:length(orth)
        Γo = orth[k].Γ
        if dir==:right
            @tensoropt (2,-2,1,-1) CL[k][i+1][-1,-2] := Γ[i][2,3,-2]*conj(Γo[i][1,3,-1])*CL[k][i][1,2]
        end
        if dir==:left
            @tensoropt (-1,-2,3,2) CR[k][i-1][-1,-2] := Γ[i][-2,1,2]*conj(Γo[i][-1,1,3])*CR[k][i][3,2]
        end
    end
end

function getHeff(mps,mpo,HL,HR,i)
    L=length(mps)
    @tensor Heff[:] := HL[i][-1,1,-4]*mpo[i].data[1,-2,-5,2]*HR[i][-3,2,-6]
    return Heff
end

function HeffMult(tensor,mposite,HL,HR)
    @tensoropt (-1,4,6,-3) temp[:] := HL[-1,1,4]* mposite.data[1,-2,5,2] *tensor[4,5,6]*HR[-3,2,6]
    return temp
end
