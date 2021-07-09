"""
    DMRG(mps_input, mpo, prec, orth=[])

Use DMRG to calculate the lowest energy eigenstate orthogonal to `orth`
"""
function DMRG(mps_input::OpenMPS, mpo::MPO, prec, orth=[]::Array{Any})
    ### input: canonical random mps
    ### output: ground state mps, ground state energy
    mps = centralize(mps_input)
    L = length(mps)
    Lorth = length(orth)
    if !(norm(mps_input) ≈ 1) || L != length(mpo)
        println("ERROR in DMRG: non-normalized MPS as input or wrong length")
        return 0
    end
    if check_LRcanonical(mps[L],:right)
        canonicity=:right
    elseif check_LRcanonical(mps[1],:left)
        canonicity=:left
    else
        println("ERROR in DMRG: use canonical mps as input")
        return 0
    end

    HL = Array{Any}(undef,L)
    HR = Array{Any}(undef,L)
    CL = Array{Any}(undef,Lorth)
    CR = Array{Any}(undef,Lorth)
    initializeHLR(mps,mpo,HL,HR)
    initializeCLR(mps,CL,CR,orth)
    Hsquared = multiplyMPOs(mpo,mpo)
    omps=OpenMPS(mps)
    E, H2 = real(expectation_value(omps, mpo)), real(expectation_value(omps, Hsquared))
    var = H2 - E^2
    println("E, var = ", E, ", ", var)
    count=1
    while var > prec && count<50
        Eprev = E
        mps = sweep(mps,mpo,HL,HR,CL,CR,prec,canonicity,orth)
        canonicity = switch_canonicity(canonicity)
        omps = OpenMPS(mps)
        E, H2 = real(expectation_value(omps,mpo)), real(expectation_value(omps, Hsquared))
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
function sweep(mps, mpo, HL, HR, CL, CR, prec,canonicity, orth=[])
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
        mpsguess = vec(mps[j])
        #HeffFun(vec) = reshape(HeffMult(reshape(vec,szmps),mpo[j],HL[j],HR[j]),prod(szmps))

        for k = 1:N_orth
            @tensoropt (1,-1,2,-3) orthTensors[k][:] := CL[k][j][1,-1]*CR[k][j][2,-3]*conj(orth[k][j][1,-2,2])
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
            evals, evecs = eigs(hefflin,nev=2,which=:SR,tol=prec,v0=mpsguess)
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
            @tensor mps[j+1][-1,-2,-3] := R[-1,1]*mps[j+1][1,-2,-3];
        elseif canonicity==:left
            @tensor mps[j-1][-1,-2,-3] := R[1,-3]*mps[j-1][-1,-2,1];
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
function eigenstates(mps, hamiltonian, prec,n)
    states = []
    energies = []
    for k = 1:n
        @time state, E = DMRG(mps,hamiltonian,prec,states)
        append!(states,[state])
        append!(energies,E)
    end
    return states, energies
end

function initializeHLR(mps,mpo,HL,HR)
    L = length(mps)

    HR[L] = Array{ComplexF64}(undef,1,1,1)
    HR[L][1,1,1] = 1
    HL[1] = Array{ComplexF64}(undef,1,1,1)
    HL[1][1,1,1] = 1

    for j=L-1:-1:1
        @tensoropt (-1,1,-3,3) HR[j][-1,-2,-3] := conj(mps[j+1][-1,4,1])*mpo[j+1].data[-2,4,5,2]*mps[j+1][-3,5,3]*HR[j+1][1,2,3]
    end
    for j=2:L
        @tensoropt (1,3,-1,-3) HL[j][-1,-2,-3] := HL[j-1][1,2,3]*conj(mps[j-1][1,4,-1])*mpo[j-1].data[2,4,5,-2]*mps[j-1][3,5,-3]
    end
end

function initializeCLR(mps,CL,CR,orth=[])
    L = length(mps)
    for k = 1:length(orth)
        CR[k] = Array{Array{ComplexF64,2}}(undef,L)
        CL[k] = Array{Array{ComplexF64,2}}(undef,L)
        CR[k][L] = Array{ComplexF64}(undef,1,1)
        CR[k][L][1,1] = 1
        CL[k][1] = Array{ComplexF64}(undef,1,1)
        CL[k][1][1,1] = 1
        for j=1:L-1
            @tensoropt (-2,2,-1,3) CR[k][L-j][-1,-2] := mps[L-j+1][-2,1,2]*conj(orth[k][L-j+1][-1,1,3])*CR[k][L-j+1][3,2]
            @tensoropt (2,-2,1,-1) CL[k][1+j][-1,-2] := mps[j][2,3,-2]*conj(orth[k][j][1,3,-1])*CL[k][j][1,2]

        end
    end
end

""" Update HL, HR, when tensor i has been updated in a dir-sweep"""
function updateHLR(mps,mpo,HL,HR,i,dir)
    L = length(mps)

    if dir==:right
        @tensoropt (1,3,-1,-3) HL[i+1][-1,-2,-3] := HL[i][1,2,3]*conj(mps[i][1,4,-1])*mpo[i].data[2,4,5,-2]*mps[i][3,5,-3]
    end
    if dir==:left
        @tensoropt (-1,1,-3,3) HR[i-1][-1,-2,-3] := conj(mps[i][-1,4,1])*mpo[i].data[-2,4,5,2]*mps[i][-3,5,3]*HR[i][1,2,3]
    end

end

function updateCLR(mps,CL,CR,i,dir,orth=[])
    L = length(mps)
    for k = 1:length(orth)
        if dir==:right
            @tensoropt (2,-2,1,-1) CL[k][i+1][-1,-2] := mps[i][2,3,-2]*conj(orth[k][i][1,3,-1])*CL[k][i][1,2]
        end
        if dir==:left
            @tensoropt (-1,-2,3,2) CR[k][i-1][-1,-2] := mps[i][-2,1,2]*conj(orth[k][i][-1,1,3])*CR[k][i][3,2]
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
