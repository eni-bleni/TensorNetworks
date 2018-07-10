module MPS
using TensorOperations
using LinearMaps
# define Pauli matrices
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]


"""
Returns the MPO for a 2-site Hamiltonian
"""
function MPOforHam(ham,L)
    d = size(ham)[1]
    mpo = Array{Any}(L)
    tmp = reshape(permutedims(ham,[1,3,2,4]),d*d,d*d)
    U,S,V = svd(tmp)
    U = reshape(U*diagm(sqrt.(S)),d,d,size(S)[1])
    V = reshape(diagm(sqrt.(S))*V',size(S)[1],d,d)
    mpo[1] = permutedims(reshape(U,d,d,size(S),1),[1,4,3,2])
    mpo[L] = permutedims(reshape(V,size(S),d,d,1),[2,1,4,3])
    @tensor begin
        tmpEven[-1,-2,-3,-4] := V[-2,-1,1]*U[1,-4,-3];
        tmpOdd[-1,-2,-3,-4] := U[-1,1,-3]*V[-2,1,-4];
    end
    for i=2:L-1
        if iseven(i)
            mpo[i] = tmpEven
        else
            mpo[i] = tmpOdd
        end
    end
    return mpo
end


""" gives the mpo corresponding to a*mpo1 + b*mpo2.
 """
function addmpos(mpo1,mpo2,reduce=true,a=1,b=1)
    L = length(mpo1)
    d= size(mpo1[1])[2]
    mpo = Array{Array{Complex{Float64}}}(L)
    mpo[1] = permutedims(cat(1,permutedims(a*mpo1[1],[4,1,2,3]),permutedims(b*mpo2[1],[4,1,2,3])),[2,3,4,1])
    for i = 2:L-1
        mpo[i] = permutedims([permutedims(mpo1[i],[1,4,2,3]) zeros(size(mpo1[i])[1],size(mpo2[i])[4],d,d); zeros(size(mpo2[i])[1],size(mpo1[i])[4],d,d) permutedims(mpo2[i],[1,4,2,3])],[1,3,4,2])
        if reduce
            @tensor tmp[-1,-2,-3,-4,-5,-6] := mpo[i-1][-1,-2,-3,1]*mpo[i][1,-4,-5,-6]
            tmp = reshape(tmp,size(mpo[i-1])[1]*d*d,d*d*size(mpo[i])[4])
            U,S,V = svd(tmp)
            V = V'
            if S[length(S)] <1e-6
                D = 0
                while (D+1 < length(S)) & (S[D+1]>1e-6)
                    D+=1
                end
                U,S,V = truncate_svd(U,S,V,D)
            else D = length(S)
            end
            mpo[i-1] = reshape(1/2*U*diagm(S),size(mpo[i-1])[1],d,d,D)
            mpo[i] = reshape(2*V,D,d,d,size(mpo[i])[4])
        end
    end
    mpo[L] = permutedims(cat(1,permutedims(mpo1[L],[1,4,2,3]),permutedims(mpo2[L],[1,4,2,3])),[1,3,4,2])
    if reduce
        @tensor tmp[-1,-2,-3,-4,-5,-6] := mpo[L-1][-1,-2,-3,1]*mpo[L][1,-4,-5,-6]
        tmp = reshape(tmp,size(mpo[L-1])[1]*d*d,d*d*size(mpo[L])[4])
        U,S,V = svd(tmp)
        V = V'
        if S[length(S)] <1e-6
            D = 0
            while (D+1 < length(S)) & (S[D+1]>1e-6)
                D+=1
            end
            U,S,V = truncate_svd(U,S,V,D)
        else D = length(S)
        end
        mpo[L-1] = reshape(1/2*U*diagm(S),size(mpo[L-1])[1],d,d,D)
        mpo[L] = reshape(2*V,D,d,d,size(mpo[L])[4])
    end
    return mpo
end

function truncate_svd(U, S, V, D)
    U = U[:, 1:D]
    S = S[1:D]
    V = V[1:D, :]
    return U, S, V
end

"""trancates the full MPS/MPO. There seems to be some bug """
function truncate2(MPSO,eps=1e-6)
    MP = MPSO
    ismpo = false
    L = length(MP)
    if length(size(MP[1])) == 4
        ismpo = true
        for i = 1:L
            s = size(MP[i])
            MP[i] = reshape(MP[i],s[1],s[2]*s[3],s[4])
        end
    end
    for i = 1:L-1
        @tensor tmp[-1,-2,-3,-4] := MP[i][-1,-2,1]*MP[i+1][1,-3,-4];
        s1 = size(MP[i]); s2 = size(MP[i+1]);
        tmp = reshape(tmp,s1[1]*s1[2],s2[2]*s2[3])
        U,S,V = svd(tmp)
        V = V'
        if S[length(S)] < eps
            D = 0
            while (D+1 < length(S)) & (S[D+1]>1e-6)
                D+=1
            end
            U,S,V = truncate_svd(U,S,V,D)
        else D = length(S)
        end
        MP[i] = reshape(1/2*U*diagm(S),s1[1],s1[2],D)
        MP[i+1] = reshape(2*V,D,s2[2],s2[3])
    end
    if ismpo
        for i=1:L
            s = size(MP[i])
            MP[i] = reshape(MP[i],s[1],round(Int,sqrt(s[2])),round(Int,sqrt(s[2])),s[3])
        end
    end
    return MP
end



""" Returns the left or right canonical form of a single tensor:
    -1 is leftcanonical, 1 is rightcanonical

        ```LRcanonical(tensor,direction) -> A,R,DB```"""
function LRcanonical(M,dir)
    D1,d,D2 = size(M); # d = phys. dim; D1,D2 = bond dims
    if dir == -1
        M = permutedims(M,[2,1,3]);
        M = reshape(M,D1*d,D2);
        A,R = qr(M); # M = Q R
        DB = size(R)[1]; # intermediate bond dimension
        A = reshape(A,d,D1,DB);
        A = permutedims(A,[2,1,3]);
    elseif dir == 1
        M = permutedims(M,[1,3,2]);
        M = reshape(M,D1,d*D2);
        A,R = qr(M');
        A = A';
        R = R';
        DB = size(R)[2];
        A = reshape(A,DB,D2,d);
        A = permutedims(A,[1,3,2]);
    else println("ERROR: not left or right canonical!");
        A = R = DB = 0;
    end
    return A,R,DB
end

""" Returns an MPO of length L for with Operators O_i at position  j_i

        ```MpoFromOperators(ops,L) -> mpo```"""
function MpoFromOperators(ops,L)
    mpo = Array{Any}(L)
    d = size(ops[1][1])[1]
    for i = 1:L
        mpo[i] = reshape(eye(d),1,d,d,1)
    end
    for i = 1:length(ops)
        mpo[ops[i][2]] = reshape(ops[i][1],1,d,d,1)
    end
    return mpo
end

""" Computes the expectation value of operators O_i sitting on site j_i

```Correlator(ops,mps) -> corr```"""
function Correlator(ops,mps)
     MPO = MpoFromOperators(ops,length(mps))
     corr = mpoExpectation(mps, MPO)
     return corr
end


"""Computes the correlation lenght for any mps by brute force calculation of
   <O_1 O_2> - <O_1>*<O_2> for distances m=1:L and physical dimension d

```correlation_length(mps, d) --> corr, xi, ind_max, a, b```"""
function correlation_length(mps, d)
    L = length(mps)
    corr = Array{Any}(L,2)
    ind_max = 1

    O1 = randn(d,d)
    O2 = randn(d,d)

    for m = 1:L # calculation of correlation function in dpendence of distance m
        ops_list = [[O1,1],[O2,m]]
        corr[m,1] = m
        if length(size(mps[1])) == 4 # MPO case
            corr[m,2] = traceMPO(multiplyMPOs(mps, MpoFromOperators(ops_list,L))) - traceMPO(multiplyMPOs(mps, MpoFromOperators([ops_list[1]],L)))*traceMPO(multiplyMPOs(mps, MpoFromOperators([ops_list[2]],L)))
        else
            corr[m,2] = MPS.Correlator(ops_list,mps) - MPS.Correlator([ops_list[1]],mps)*MPS.Correlator([ops_list[2]],mps)
        end
    end

    for m = 1:L # calculation of maximal index up to which corr is above machine precision (for fit interval)
        if abs(corr[m,2]) > 1e-14
            ind_max = m
        else
            break
        end
    end
    # println("ind_max: ", ind_max)

    a, b = linreg(corr[1:ind_max,1], log.(abs.(corr[1:ind_max,2])))
    xi = -1/b
    # println("xi = ", xi)

    return corr, xi, ind_max, a, b
end


""" Returns the Identity chain as an MPO

```IdentityMPO(lattice sites, phys dims)```"""
function IdentityMPO(L, d)
    # mpo = Array{Array{Complex{Float32},4}}(L)
    mpo = Array{Any}(L)
    for i = 1:L
        mpo[i] = Array{Complex128}(1,d,d,1)
        mpo[i][1,:,:,1] = eye(d)
    end

    return mpo
end


""" Returns the Hamiltonian for the Ising model in transverse field as an MPO

```IsingMPO(lattice sites,J,transverse,longitudinal)```"""
function IsingMPO(L, J, h, g, shift=0)
    ### input:   L: lenght of mpo = number of sites/tensors; J,h,g: Ising Hamiltonian params
    ### constructs Hamiltonian sites of size (a,i,j,b) -> a,b: bond dims, i,j: phys dims
    ### first site: (i,j,b); last site: (a,i,j)
    mpo = Array{Any}(L)
    mpo[1] = Array{Complex128}(1,2,2,3)
    mpo[1][1,:,:,:] = reshape([si J*sz h*sx+g*sz+shift*si/L],2,2,3)
    mpo[L] = Array{Complex128}(3,2,2,1)
    mpo[L][:,:,:,1] = permutedims(reshape([h*sx+g*sz+shift*si/L sz si], 2,2,3), [3,1,2])
    for i=2:L-1
        # hardcoded implementation of index structure (a,i,j,b):
        help = Array{Complex128}(3,2,2,3)
        help[1,:,:,1] = help[3,:,:,3] = si
        help[1,:,:,2] = J*sz
        help[1,:,:,3] = h*sx+g*sz+shift*si/L
        help[2,:,:,1] = help[2,:,:,2] = help[3,:,:,1] = help[3,:,:,2] = s0
        help[2,:,:,3] = sz
        mpo[i] = help
    end
    return mpo
end

""" Returns the Hamiltonian for the Heisenberg model in transverse field as an MPO

```HeisenbergMPO(lattice sites,Jx,Jy,Jz,transverse)```"""
function HeisenbergMPO(L, Jx, Jy, Jz, h)
    ### input: L:          lenght of mpo = number of sites/tensors;
    ###        Jx,Jy,Jz,h: params for the quantum Heisenberg Hamiltonian
    ###        H = sum_i( Jx*sx_{i}*sx_{i+1} + Jy*sy_{i}*sy_{i+1} + Jz*sz_{i}*sz_{i+1} + h*sx_{i} )
    ### output:
    ### constructs Hamiltonian sites of size (a,i,j,b) -> a,b: bond dims, i,j: phys dims
    ###                          first site: (i,j,b); last site: (a,i,j)

    mpo = Array{Any}(L)
    mpo[1] = Array{Complex128}(1,2,2,5)
    mpo[1][1,:,:,:] = reshape([si Jx*sx Jy*sy Jz*sz h*sx], 2,2,5)
    mpo[L] = Array{Complex128}(5,2,2,1)
    mpo[L][:,:,:,1] = permutedims(reshape([h*sx sx sy sz si], 2,2,5), [3,1,2])

    for i=2:L-1
        # hardcoded implementation of index structure (a,i,j,b):
        help = Array{Complex128}(5,2,2,5)
        help[1,:,:,1] = help[5,:,:,5] = si
        help[1,:,:,2] = Jx*sx
        help[1,:,:,3] = Jy*sy
        help[1,:,:,4] = Jz*sz
        help[1,:,:,5] = h*sx
        help[2,:,:,5] = sx
        help[3,:,:,5] = sy
        help[4,:,:,5] = sz
        #help[2,:,:,1:4] = help[3,:,:,1:4] = help[4,:,:,1:4] = help[5,:,:,1:4] = s0
        help[2,:,:,1] = help[2,:,:,2] = help[2,:,:,3] = help[2,:,:,4] = s0
        help[3,:,:,1] = help[3,:,:,2] = help[3,:,:,3] = help[3,:,:,4] = s0
        help[4,:,:,1] = help[4,:,:,2] = help[4,:,:,3] = help[4,:,:,4] = s0
        help[5,:,:,1] = help[5,:,:,2] = help[5,:,:,3] = help[5,:,:,4] = s0
        mpo[i] = help
    end
    return mpo
end


""" Returns a random MPS

```randomMPS(length,physical dim, bond dim)```"""
function randomMPS(L,d,D)
    ### L: lenght of mps = number of sites/tensors
    ### d: physical dim
    ### D: bond dim

    mps = Array{Any}(L)
    ran = rand(d,D)+1im*rand(d,D)
    ran = ran/norm(ran)
    mps[1] = reshape(ran,1,d,D)
    for i = 2:1:L-1
        ran = rand(D,d*D)+1im*rand(D,d*D)
        ran = ran/norm(ran)
        mps[i] = reshape(ran,D,d,D)
    end
    ran = rand(D,d)+1im*rand(D,d)
    ran = ran/norm(ran)
    mps[L] = reshape(ran,D,d,1)
    return mps
end


""" employs the variational MPS method to find the ground state/energy.
    The state will be orthogonal to orth (optional argument).
    ```DMRG(mps,hamiltonian mpo,precision,orth=nothing) -> mps, energy```"""
function DMRG(mps_input, mpo, prec, orth=[])
    ### input: canonical random mps
    ### output: ground state mps, ground state energy

    mps = 1*mps_input  # ATTENTION: necessary trick to keep mps local variable
    L = length(mps)
    Lorth = length(orth)
    if !(MPSnorm(mps) ≈ 1) | L != length(mpo)
        println("ERROR in DMRG: non-normalized MPS as input or wrong length")
        return 0
    end
    if check_LRcanonical(mps[L],1)
        canonicity=1
    elseif check_LRcanonical(mps[1],-1)
        canonicity=-1
    else
        println("ERROR in DMRG: use canonical mps as input")
        return 0
    end

    HL = Array{Any}(L)
    HR = Array{Any}(L)
    CL = Array{Any}(Lorth)
    CR = Array{Any}(Lorth)
    initializeHLR(mps,mpo,HL,HR)
    initializeCLR(mps,CL,CR,orth)

    E, H2 = real(mpoExpectation(mps,mpo)), real(mpoSquaredExpectation(mps,mpo))
    var = H2 - E^2
    println("E, var = ", E, ", ", var)
    count=1
    while var > prec && count<50
        Eprev = E
        mps, E, var, canonicity = sweep(mps,mpo,HL,HR,CL,CR,prec,canonicity,orth)
        println("E, var = ", E, ", ", var)
        count=count+1
        if abs((Eprev-E)/E) < prec
            break
        end
    end

    return mps, E
end


""" sweeps from left to right in the DMRG algorithm """
function sweep(mps, mpo, HL, HR, CL, CR, prec,canonicity, orth=[])
    ### minimizes E by diagonalizing site by site in the mps from left to right: j=1-->L-1
    ### the resulting sites are left-canonicalized
    L = length(mps)
    for j = 1:L-1
        if canonicity==-1
            j=L+1-j
        end

        szmps = size(mps[j])
        mpsguess = reshape(mps[j],prod(szmps))
        HeffFun(vec) = reshape(HeffMult(reshape(vec,szmps),mpo[j],HL[j],HR[j]),prod(szmps))
        hefflin = LinearMap{Complex128}(HeffFun, prod(szmps),ishermitian=true)
        proj = 1
        for k = 1:length(orth)
            @tensor orthTensor[:] := CL[k][j][1,-1]*CR[k][j][2,-3]*conj(orth[k][j][1,-2,2])
            so = size(orthTensor)
            orthvector = reshape(orthTensor,1,prod(so))
            orthvector = orthvector/norm(orthvector)
            tmp = [zeros(prod(so)) nullspace(orthvector)]
            proj = LinearMap(proj*LinearMap(tmp')*LinearMap(tmp),ishermitian=true)
        end
        if orth!=nothing
            hefflin = LinearMap(proj * hefflin * proj',ishermitian=true)
            mpsguess = proj*mpsguess
        end

        if size(hefflin)[1] < 10
            evals, evecs = eig(Base.full(hefflin))
        else
            evals, evecs = eigs(hefflin,nev=2,which=:SR,tol=prec,v0=mpsguess)
        end

        if !(evals ≈ real(evals))
            println("ERROR: no real eigenvalues")
            return 0
        end
        evals = real(evals)
        eval_min, ind_min = minimum(evals), indmin(evals)
        evec_min = evecs[:,ind_min]

        evec_min = proj'*evec_min

        Mj = reshape(evec_min,szmps)
        Aj,R = LRcanonical(Mj,-canonicity)
        mps[j] = Aj

        if canonicity==1
            @tensor mps[j+1][-1,-2,-3] := R[-1,1]*mps[j+1][1,-2,-3];
        elseif canonicity==-1
            @tensor mps[j-1][-1,-2,-3] := R[1,-3]*mps[j-1][-1,-2,1];
        end
        updateCLR(mps,CL,CR,j,canonicity,orth)
        updateHLR(mps,mpo,HL,HR,j,canonicity)

    end

    ## Energies:
    E, H2 = mpoExpectation(mps,mpo), mpoSquaredExpectation(mps,mpo)
    if isapprox(E,real(E); atol = prec)  &&  isapprox(H2,real(H2); atol=prec)
        E, H2 = real(E), real(H2)
    else
        println("ERROR: no real energies")
        return 0
    end
    var = H2 - E^2
    return mps, E, var, -canonicity
end


function n_lowest_states(mps, hamiltonian, prec,n)
    states = []
    energies = []
    for k = 1:n
        @time state,E = MPS.DMRG(mps,hamiltonian,prec,states)
        append!(states,[state])
        append!(energies,E)
    end
    return states,energies
end


function initializeHLR(mps,mpo,HL,HR)
    L = length(mps)

    HR[L] = Array{Complex128}(1,1,1)
    HR[L][1,1,1] = 1
    HL[1] = Array{Complex128}(1,1,1)
    HL[1][1,1,1] = 1

    for j=L-1:-1:1
        @tensor HR[j][-1,-2,-3] := conj(mps[j+1][-1,4,1])*mpo[j+1][-2,4,5,2]*mps[j+1][-3,5,3]*HR[j+1][1,2,3]
    end
    for j=2:L
        @tensor HL[j][-1,-2,-3] := HL[j-1][1,2,3]*conj(mps[j-1][1,4,-1])*mpo[j-1][2,4,5,-2]*mps[j-1][3,5,-3]
    end
end

function initializeCLR(mps,CL,CR,orth=[])
    L = length(mps)

    for k = 1:length(orth)
        CR[k] = Array{Array{Complex128,2}}(L)
        CL[k] = Array{Array{Complex128,2}}(L)
        CR[k][L] = Array{Complex128}(1,1)
        CR[k][L][1,1] = 1
        CL[k][1] = Array{Complex128}(1,1)
        CL[k][1][1,1] = 1
        for j=1:L-1
            @tensor begin
                CR[k][L-j][-1,-2] := mps[L-j+1][-2,1,2]*conj(orth[k][L-j+1][-1,1,3])*CR[k][L-j+1][3,2]
                CL[k][1+j][-1,-2] := mps[j][2,3,-2]*conj(orth[k][j][1,3,-1])*CL[k][j][1,2]
            end
        end
    end
end

""" Update HL, HR, when tensor i has been updated in a dir-sweep"""
function updateHLR(mps,mpo,HL,HR,i,dir)
    L = length(mps)
    if dir==1
        @tensor HL[i+1][-1,-2,-3] := HL[i][1,2,3]*conj(mps[i][1,4,-1])*mpo[i][2,4,5,-2]*mps[i][3,5,-3]
    end
    if dir==-1
        @tensor HR[i-1][-1,-2,-3] := conj(mps[i][-1,4,1])*mpo[i][-2,4,5,2]*mps[i][-3,5,3]*HR[i][1,2,3]
    end
end

function updateCLR(mps,CL,CR,i,dir,orth=[])
    L = length(mps)
    for k = 1:length(orth)
        if dir==1
            @tensor CL[k][i+1][-1,-2] := mps[i][2,3,-2]*conj(orth[k][i][1,3,-1])*CL[k][i][1,2]
        end
        if dir==-1
            @tensor CR[k][i-1][-1,-2] := mps[i][-2,1,2]*conj(orth[k][i][-1,1,3])*CR[k][i][3,2]
        end
    end
end

function getHeff(mps,mpo,HL,HR,i)
    L=length(mps)
    @tensor Heff[:] := HL[i][-1,1,-4]*mpo[i][1,-2,-5,2]*HR[i][-3,2,-6]
    return Heff
end

function multiplyMPOs(mpo1,mpo2; c=true)
    L = length(mpo1)
    mpo = Array{Any}(L)
    for j=1:L
        if c
            @tensor temp[:] := mpo1[j][-1,-3,1,-5] * conj(mpo2[j][-2,-4,1,-6])
        else
            @tensor temp[:] := mpo1[j][-1,-3,1,-5] * mpo2[j][-2,1,-4,-6]
        end
        s=size(temp)
        mpo[j] = reshape(temp,s[1]*s[2],s[3],s[4],s[5]*s[6])
    end

    return mpo
end


"""
calculates Tr(mpo^n) for n=1,2,4
"""
function traceMPO(mpo,n=1)
    L = length(mpo)
    if n == 1
        F = Array{Complex64}(1,1)
        F[1,1] = 1
        for i = 1:L
            @tensor F[-1,-2] := F[-1,1]*mpo[i][1,2,2,-2]
        end
        return F[1,1]
    elseif n == 2
        F = Array{Complex64}(1,1,1,1)
        F[1,1,1,1] = 1
        for i = 1:L
            @tensor F[-1,-2,-3,-4] := F[-1,-2,1,2]*mpo[i][1,3,4,-3]*conj(mpo[i][2,3,4,-4])
        end
        return F[1,1,1,1]
    elseif n == 4
        F = Array{Complex64}(1,1,1,1,1,1,1,1)
        F[1,1,1,1,1,1,1,1] = 1
        for i = 1:L
            @tensor F[-1,-2,-3,-4,-5,-6,-7,-8] := F[-1,-2,-3,-4,1,2,3,4]*mpo[i][1,5,6,-5]*conj(mpo[i][2,7,6,-6])*conj(mpo[i][3,8,7,-7])*mpo[i][4,8,5,-8]
        end
        return F[1,1,1,1,1,1,1,1]
    else
        println("ERROR: choose n=1,2,4 in traceMPO(mpo,n=1)")
        return "nan"
    end
end


"""
calculates Tr(mpo1^n * mpo2) for n=1,2
"""
function traceMPOprod(mpo1,mpo2,n=1)
    L = length(mpo1)
    if n == 1
        F = Array{Complex64}(1,1,1,1)
        F[1,1,1,1] = 1
        for i = 1:L
            @tensor F[-1,-2,-3,-4] := F[-1,-2,1,2]*mpo1[i][1,3,4,-3]*conj(mpo2[i][2,3,4,-4])
        end
        return F[1,1,1,1]
    elseif n == 2
        F = Array{Complex64}(1,1,1,1,1,1)
        F[1,1,1,1,1,1] = 1
        for i = 1:L
            @tensor F[-1,-2,-3,-4,-5,-6] := F[-1,-2,-3,1,2,3]*mpo1[i][1,4,5,-4]*conj(mpo1[i][2,6,5,-5])*mpo2[i][3,6,4,-6]
        end
        return F[1,1,1,1,1,1]
    else
        println("ERROR: choose n=1,2 in traceMPOprod(mpo1,mpo2,n=1)")
        return "nan"
    end
end

function HeffMult(tensor,mpo,HL,HR)
    @tensor temp[:] := HL[-1,1,4]*(mpo[1,-2,5,2]*tensor[4,5,6])*HR[-3,2,6]
    return temp
end


""" returns the mpo expectation value <mps1|mpo|mps2>

    ```mpoExpectation(mps1,mpo,mps2=mps1)```"""
function mpoExpectation(mps1,mpo,mps2=mps1)
    L = length(mps1)
    if L != length(mpo)
        println("ERROR: MPS and MPO do not have same length")
        return 0
    end
    F = Array{Complex128}(1,1,1)
    F[1,1,1] = 1
    for i = 1:L
        @tensor F[-1,-2,-3] := F[1,2,3]*mps1[i][3,5,-3]*mpo[i][2,4,5,-2]*conj(mps2[i][1,4,-1])
    end
    return F[1,1,1]
end


""" returns the squared mpo expectation value <mps|mpo^2|mps>

    ```mpoSquaredExpectation```"""
function mpoSquaredExpectation(mps, mpo)
    L = length(mps)
    if L != length(mpo)
        println("ERROR: MPS and MPO do not have same length")
        return 0
    end
    F = Array{Complex128}(1,1,1,1)
    F[1,1,1,1] = 1
    for i = 1:L
       @tensor F[-1,-2,-3,-4] := F[1,2,3,4]*mps[i][4,5,-4]*mpo[i][3,6,5,-3]*mpo[i][2,4,6,-2]*conj(mps[i][1,4,-1])
    end
    return F[1,1,1,1]
end

""" returns the norm of an MPS """
function MPSnorm(mps)
    C = Array{Complex128}(1,1)
    C[1,1] = 1
    for i=1:length(mps)
        @tensor C[-1,-2] := mps[i][2,3,-2]*C[1,2]*conj(mps[i][1,3,-1])
    end
    return C[1,1]
end

function MPSoverlap(mps1,mps2)
    if length(mps1) != length(mps2)
        println("ERROR: the two MPS are not of the same size")
        return 0
    else
        L = length(mps1)
        C = Array{Complex128}(1,1)
        C[1,1] = 1
        for i=1:L
            @tensor C[-1,-2] := mps1[i][1,3,-2]*C[2,1]*conj(mps2[i][2,3,-1])
        end
    end
    return C[1,1]/(sqrt.(real(MPSnorm(mps1)*MPSnorm(mps2))))
end

""" check whether given site/tensor in mps is left-/rightcanonical
    ```check_LRcanonical(tensor,dir)```"""
function check_LRcanonical(a, dir)
    ### this function checks if a*a' contracts to the kronecker/identity matrix
    ### dir: -1=leftcanonical, 1=rightcanonical
    ### output: boolean true or false
    if dir == -1    # check if site is leftcanonical
        @tensor c[-1,-2] := a[1,2,-2]*conj(a[1,2,-1])
    elseif dir == 1 # check if site is rightcanonical
        @tensor c[-1,-2] := a[-1,2,1]*conj(a[-2,2,1])
    else
        println("ERROR: choose -1 for leftcanonical or +1 for rightcanonical")
        return false
    end
    return c ≈ eye(size(c)[1])
end


""" Make mps L-can left of site n and R-can right of site.
    No site specified implies right canonical

    ``` makeCanonical(mps,n=0)```"""
function makeCanonical(mps,n=0)
    L = length(mps)
    mpo_trafo = 0

    if length(size(mps[1])) == 4 # make mps out of mpo
        mpo_trafo = 1
        smps = Array{Any}(L)
        for i = 1:L
            smps[i] = size(mps[i])
            mps[i] = reshape(mps[i], smps[i][1],smps[i][2]*smps[i][3],smps[i][4])
        end
    end

    for i = 1:n-1
        mps[i],R,DB = LRcanonical(mps[i],-1);
        if i<L
            @tensor mps[i+1][-1,-2,-3] := R[-1,1]*mps[i+1][1,-2,-3];
        end
    end
    for i = L:-1:n+1
        mps[i],R,DB = LRcanonical(mps[i],1);
        if i>1
            @tensor mps[i-1][:] := mps[i-1][-1,-2,1]*R[1,-3]
        end
    end

    if mpo_trafo == 1 # transform mps back into mpo
        for i = 1:L
            mps[i] = reshape(mps[i], smps[i][1],smps[i][2],smps[i][3],smps[i][4])
        end
    end
end



"""
    ```mpo_to_mps(mpo)```"""
function mpo_to_mps(mpo)
    L = length(mpo)
    smpo = Array{Any}(L)
    for i = 1:L
        smpo[i] = size(mpo[i])
        mpo[i] = reshape(mpo[i], smpo[i][1],smpo[i][2]*smpo[i][3],smpo[i][4])
    end

    return smpo
end


"""
    ```mps_to_mpo(mps, smpo)```"""
function mps_to_mpo(mps, smpo)
    ## smpo[i] = size(mpo[i])
    L = length(mps)
    for i = 1:L
        mps[i] = reshape(mps[i], smpo[i][1],smpo[i][2],smpo[i][3],smpo[i][4])
    end
end


"""
constructs |Psi><Psi| as an MPO
"""
function pureDensityMatrix(mps)
    L = length(mps)
    rho = Array{Any}(L)
    for i = 1:L
        D1,d,D2 = size(mps[i])
        @tensor help[-1,-2,-3,-4,-5,-6] := conj(mps[i][-3,-1,-2])*mps[i][-5,-6,-4]
        help = reshape(permutedims(help, [3,5,6,1,2,4]), D1*D1,d,d,D2*D2)
        rho[i] = help
    end

    return rho
end


""" UNFINISHED. Von Neumann entropy across link i
``` entropy(mps,i) -> S```"""
function entropy(mps,i)
    makeCanonical(mps,i+1)
    sz = size(mps[i+1])
    tensor = reshape(mps[i+1],sz[1],sz[2]*sz[3])
    U,S,V = svd(tensor)
    S = S.^2
    return -dot(S,log.(S))
end


"""
Subsystem (1,l)<(1,L) squared trace distance btw MPO and MPS
"""
function SubTraceDistance(MPO,MPS,l)
    L = length(MPO)
    A = Array{Complex64}(1,1)
    B1 = Array{Complex64}(1,1,1)
    B2 = Array{Complex64}(1,1,1)
    C = Array{Complex64}(1,1,1,1)
    A[1,1] = 1
    B1[1,1,1] = 1
    B2[1,1,1] = 1
    C[1,1,1,1] = 1
    for i=1:l
        @tensor begin
                    A[-1,-2] := A[1,2]*MPO[i][1,4,3,-1]*conj(MPO[i][2,4,3,-2])
                    B1[-1,-2,-3] := B1[1,2,3]*MPS[i][1,4,-1]*MPO[i][2,4,5,-2]*conj(MPS[i][3,5,-3])
                    B2[-1,-2,-3] := B2[1,2,3]*MPS[i][1,4,-1]*conj(MPO[i][2,5,4,-2])*conj(MPS[i][3,5,-3])
                    C[-1,-2,-3,-4] := C[1,2,3,4]*conj(MPS[i][1,5,-1])*MPS[i][2,6,-2]*conj(MPS[i][3,6,-3])*MPS[i][4,5,-4]
                end
    end
    for i=l+1:L
        @tensor begin
                    A[-1,-2] := A[1,2]*MPO[i][1,3,3,-1]*conj(MPO[i][2,4,4,-2])
                    B1[-1,-2,-3] := B1[1,2,3]*MPS[i][1,4,-1]*MPO[i][2,5,5,-2]*conj(MPS[i][3,4,-3])
                    B2[-1,-2,-3] := B2[1,2,3]*MPS[i][1,4,-1]*conj(MPO[i][2,5,5,-2])*conj(MPS[i][3,4,-3])
                    C[-1,-2,-3,-4] := C[1,2,3,4]*conj(MPS[i][1,5,-1])*MPS[i][2,5,-2]*conj(MPS[i][3,6,-3])*MPS[i][4,6,-4]
                end
    end
    return A[1,1] - B1[1,1,1] - B2[1,1,1] + C[1,1,1,1]
end


end
