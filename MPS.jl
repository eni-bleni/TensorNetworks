module MPS
using TensorOperations

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

# define Pauli matrices
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]


""" OneSiteMPO(L, j, op)
returns a MPO of length L with identities at each site and operator 'op' at site j """
function OneSiteMPO(L, j, op)
    mpo = Array{Any}(L)
    for i = 1:L
        mpo[i] = reshape(si, 1,2,2,1)
    end
    mpo[j] = reshape(op, 1,2,2,1)

    return mpo
end


""" Returns the Hamiltonian for the Ising model in transverse field as an MPO

```IsingMPO(lattice sites,J,transverse,longitudinal)```"""
function IsingMPO(L, J, h, g)
    ### input:   L: lenght of mpo = number of sites/tensors; J,h,g: Ising Hamiltonian params
    ### constructs Hamiltonian sites of size (a,i,j,b) -> a,b: bond dims, i,j: phys dims
    ### first site: (i,j,b); last site: (a,i,j)
    mpo = Array{Any}(L)
    mpo[1] = Array{Complex64}(1,2,2,3)
    mpo[1][1,:,:,:] = reshape([si J*sz h*sx+g*sz],2,2,3)
    mpo[L] = Array{Complex64}(3,2,2,1)
    mpo[L][:,:,:,1] = permutedims(reshape([h*sx+g*sz sz si], 2,2,3), [3,2,1])
    for i=2:L-1
        # hardcoded implementation of index structure (a,i,j,b):
        help = Array{Complex64}(3,2,2,3)
        help[1,:,:,1] = help[3,:,:,3] = si
        help[1,:,:,2] = J*sz
        help[1,:,:,3] = h*sx+g*sz
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
    mpo[1] = Array{Complex64}(1,2,2,5)
    mpo[1][1,:,:,:] = reshape([si Jx*sx Jy*sy Jz*sz h*sx], 2,2,5)
    mpo[L] = Array{Complex64}(5,2,2,1)
    mpo[L][:,:,:,1] = permutedims(reshape([h*sx sx sy sz si], 2,2,5), [3,2,1])

    for i=2:L-1
        # hardcoded implementation of index structure (a,i,j,b):
        help = Array{Complex64}(5,2,2,5)
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

    ```DMRG(mps,hamiltonian mpo,precision,orth=nothing)```"""
function DMRG(mps_input, mpo, prec, orth=nothing)
    ### input: canonical random mps
    ### output: ground state mps, ground state energy

    mps = 1*mps_input  # ATTENTION: necessary trick to keep mps local variable
    L = length(mps)

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
    CL = Array{Any}(L)
    CR = Array{Any}(L)
    initializeHLR(mps,mpo,HL,HR)
    initializeCLR(mps,CL,CR,orth)

    E, H2 = real(mpoExpectation(mps,mpo)), real(mpoSquaredExpectation(mps,mpo))
    var = H2 - E^2
    println("E, var = ", E, ", ", var)
    count=1
    while var > prec && count<50
        mps, E, var, canonicity = sweep(mps,mpo,HL,HR,CL,CR,prec,canonicity,orth)
        println("E, var = ", E, ", ", var)
        count=count+1
    end

    return mps, E
end

""" sweeps from left to right in the DMRG algorithm """
function sweep(mps, mpo, HL, HR, CL, CR, prec,canonicity, orth=nothing)
    ### minimizes E by diagonalizing site by site in the mps from left to right: j=1-->L-1
    ### the resulting sites are left-canonicalized
    L = length(mps)

    for j = 1:L-1
        if canonicity==-1
            j=L+1-j
        end

        Heff = getHeff(mps,mpo,HL,HR,j)
        D1,d,D2 = size(Heff)

        Heff = permutedims(Heff, [2,1,3,5,4,6])       # = (d,D1,D2, d,D1,D2)
        Heff = reshape(Heff, d*D1*D2, d*D1*D2)        # = (d*D1*D2, d*D1*D2)
        szmps = size(mps[j])
        mpsguess = reshape(permutedims(mps[j],[2,1,3]),szmps[1]*szmps[2]*szmps[3])

        if orth!=nothing
            @tensor orthTensor[-2,-1,-3] := CL[j][1,-1]*CR[j][2,-3]*conj(orth[j][1,-2,2])
            so = size(orthTensor)
            orthvector = reshape(orthTensor,1,so[1]*so[2]*so[3])
            orthvector = orthvector/norm(orthvector)
            proj = [zeros(orthvector') nullspace(orthvector)]'
            Heff = proj * Heff * proj'
            mpsguess = proj'*mpsguess
        end

        evals, evecs = eigs(Heff,nev=1,which=:SR,tol=prec,v0=mpsguess)
        if !(evals ≈ real(evals))
            println("ERROR: no real eigenvalues")
            return 0
        end
        evals = real(evals)
        eval_min, ind_min = minimum(evals), indmin(evals)
        evec_min = evecs[:,ind_min]

        if orth!=nothing
            evec_min = proj'*evec_min
        end

        Mj = reshape(evec_min, 2,D1,D2) # = (d,D1,D2)
        Mj = permutedims(Mj, [2,1,3])   # = (D1,d,D2)
        Aj,R = LRcanonical(Mj,-canonicity)
        mps[j] = Aj

        if canonicity==1
            @tensor mps[j+1][-1,-2,-3] := R[-1,1]*mps[j+1][1,-2,-3];
        elseif canonicity==-1
            @tensor mps[j-1][-1,-2,-3] := R[1,-3]*mps[j-1][-1,-2,1];
        end
        updateCLR(mps,CL,CR,j,canonicity,orth)
        updateHeff(mps,mpo,HL,HR,j,canonicity)

    end

    ## Energies:
    E, H2 = mpoExpectation(mps,mpo), mpoSquaredExpectation(mps,mpo)
    if (E ≈ real(E))  &  (H2 ≈ real(H2))
        E, H2 = real(E), real(H2)
    else
        println("ERROR: no real energies")
        return 0
    end
    var = H2 - E^2
    return mps, E, var, -canonicity
end

function initializeHLR(mps,mpo,HL,HR)
    L = length(mps)

    HR[L] = Array{Complex64}(1,1,1)
    HR[L][1,1,1] = 1
    HL[1] = Array{Complex64}(1,1,1)
    HL[1][1,1,1] = 1

    for j=L-1:-1:1
        @tensor HR[j][-1,-2,-3] := conj(mps[j+1][-1,4,1])*mpo[j+1][-2,4,5,2]*mps[j+1][-3,5,3]*HR[j+1][1,2,3]
    end
    for j=2:L
        @tensor HL[j][-1,-2,-3] := HL[j-1][1,2,3]*conj(mps[j-1][1,4,-1])*mpo[j-1][2,4,5,-2]*mps[j-1][3,5,-3]
    end
end

function initializeCLR(mps,CL,CR,orth=nothing)
    if orth==nothing
        return
    end
    L = length(mps)
    CR[L] = Array{Complex64}(1,1)
    CR[L][1,1] = 1
    CL[1] = Array{Complex64}(1,1)
    CL[1][1,1] = 1
    for j=1:L-1
        @tensor begin
            CR[L-j][-1,-2] := mps[L-j+1][-2,1,2]*conj(orth[L-j+1][-1,1,3])*CR[L-j+1][3,2]
            CL[1+j][-1,-2] := mps[j][2,3,-2]*conj(orth[j][1,3,-1])*CL[j][1,2]
        end
    end
end

""" Update HL, HR, when tensor i has been updated in a dir-sweep"""
function updateHeff(mps,mpo,HL,HR,i,dir)
    L = length(mps)
    if dir==1
        @tensor HL[i+1][-1,-2,-3] := HL[i][1,2,3]*conj(mps[i][1,4,-1])*mpo[i][2,4,5,-2]*mps[i][3,5,-3]
    end
    if dir==-1
        @tensor HR[i-1][-1,-2,-3] := conj(mps[i][-1,4,1])*mpo[i][-2,4,5,2]*mps[i][-3,5,3]*HR[i][1,2,3]
    end
end
function updateCLR(mps,CL,CR,i,dir,orth=nothing)
    if orth==nothing
        return
    end
    L = length(mps)
    if dir==1
        @tensor CL[i+1][-1,-2] := mps[i][2,3,-2]*conj(orth[i][1,3,-1])*CL[i][1,2]
    end
    if dir==-1
        @tensor CR[i-1][-1,-2] := mps[i][-2,1,2]*conj(orth[i][-1,1,3])*CR[i][3,2]
    end
end

function getHeff(mps,mpo,HL,HR,i)
    L=length(mps)
    @tensor Heff[:] := HL[i][-1,1,-4]*mpo[i][1,-2,-5,2]*HR[i][-3,2,-6]
    return Heff
end

""" constructs the effective Hamiltonian = environment around site j """
function constr_Heff(mps, mpo, j)
    L = length(mps)

    if j == 1 # first effective Hamiltonian = environment with blanks at mps[1]

         # start all contractions from right side
        @tensor   r[-1,-2,-3] := mps[L][-1,4,1]*mpo[L][-2,4,5]*conj(mps[L][-3,5,1])

        for i = L-1:-1:2
            @tensor begin
                r[-1,-2,-3] := mps[i][-1,4,1]*mpo[i][-2,4,5,2]*conj(mps[i][-3,5,3])*r[1,2,3]
            end
        end
        @tensor begin
            Heff[-1,-2,-3,-4] := mpo[1][-2,-3,2]*r[-1,2,-4]
        end
    elseif 1<j<L # j-th effective hamiltonian: contract blocks from left and right
        @tensor begin
            l[-1,-2,-3] := mps[1][1,2,-1]*mpo[1][2,3,-2]*conj(mps[1][1,3,-3])
            r[-1,-2,-3] := mps[L][-1,2,1]*mpo[L][-2,2,3]*conj(mps[L][-3,3,1])
        end
        if j > 2               # left block becomes only larger for j>2
            for i = 2:j-1      # contract block to the left of site j
                @tensor begin
                    l[-1,-2,-3] := l[1,2,3]*mps[i][1,4,-1]*mpo[i][2,4,5,-2]*conj(mps[i][3,5,-3])
                end
            end
        end
        if j < L-1             # right block becomes only larger for j<L-1
            for i = L-1:-1:j+1 # contract block to the right of site j
                @tensor begin
                    r[-1,-2,-3] := mps[i][-1,4,1]*mpo[i][-2,4,5,2]*conj(mps[i][-3,5,3])*r[1,2,3]
                end
            end
        end
        @tensor begin
            Heff[-1,-2,-3,-4,-5,-6] := l[-1,2,-4]*mpo[j][2,-2,-5,3]*r[-3,3,-6]
        end
    elseif j == L # last effective Hamiltonian = environment with blanks at mps[L]
        @tensor begin # start all contractions from left side
            l[-1,-2,-3] := mps[1][1,4,-1]*mpo[1][4,5,-2]*conj(mps[1][1,5,-3])
        end
        for i = 2:L-1
            @tensor begin
                l[-1,-2,-3] := l[1,2,3]*mps[i][1,4,-1]*mpo[i][2,4,5,-2]*conj(mps[i][3,5,-3])
            end
        end
        @tensor begin
            Heff[-1,-2,-3,-4] := l[-1,2,-4]*mpo[L][2,-2,-3]
        end
    else
        println("ERROR in constr_Heff: j not identified")
    end

    return Heff
end

""" returns the mpo expectation value <mps|mpo|mps>

    ```mpoExpectation(mps,mpo)```"""
function mpoExpectation(mps, mpo)
    L = length(mps)
    if L != length(mpo)
        println("ERROR: MPS and MPO do not have same length")
        return 0
    end
    F = Array{Complex64}(1,1,1)
    F[1,1,1] = 1
    for i = 1:L
        @tensor F[-1,-2,-3] := F[1,2,3]*mps[i][3,5,-3]*mpo[i][2,4,5,-2]*conj(mps[i][1,4,-1])
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
    F = Array{Complex64}(1,1,1,1)
    F[1,1,1,1] = 1
    for i = 1:L
        @tensor F[-1,-2,-3,-4] := F[1,2,3,4]*mps[i][4,5,-4]*mpo[i][2,5,6,-2]*mpo[i][3,6,7,-3]*conj(mps[i][1,7,-1])
    end
    return F[1,1,1,1]
end

""" returns the norm of an MPS """
function MPSnorm(mps)
    C = Array{Complex64}(1,1)
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
        C = Array{Complex64}(1,1)
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
end


""" UNFINISHED. Von Neumann entropy across link i
``` entropy(mps,i) -> S```"""
function entropy(mps,i)
    makeCanonical(mps)
    sz = size(mps[i+1])
    tensor = reshape(mps[i+1],sz[1],sz[2]*sz[3])
    U,S,V = svd(tensor)
    # S = S.^2/dot(S,S)
    println(norm(S))
    return -dot(S,log.(S))

end

end
