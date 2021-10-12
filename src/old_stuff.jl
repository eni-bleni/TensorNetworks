function canonicalize_one_site_deg(A,l,Dmax=10,tol=1e-8)
    TR = transfer_right(l,A)
    TL = transfer_left(A,l)
    if size(TL,1)<10
        vals, vecs = eigen(Matrix(TL))
    else
        vals, vecs = eigs(TL,nev=1)
    end
    valR = vals[end]
    nr = length(val .== valR)
    rhoR = vecs[:,end:end-nr+1]
    if size(TR,1)<10
        vals, vecs = eigen(Matrix(TR))
    else
        vals, vecs = eigs(TR,nev=1)
    end

    valL = vals[end]
    rhoL = vecs[:,end]
    println(valL[1]," :L _ R: ", valR[1])
    chi = Int(sqrt(length(rhoR)))
    rhoR = reshape(rhoR,chi,chi)
    phase = tr(rhoR)/abs(tr(rhoR))
    rhoR = rhoR ./ phase
    rhoR = Hermitian(0.5*(rhoR + rhoR'))
    chi = Int(sqrt(length(rhoL)))
    rhoL = reshape(rhoL,chi,chi)
    phase = tr(rhoL)/abs(tr(rhoL))
    rhoL = rhoL ./ phase
    rhoL = Hermitian(.5*(rhoL + rhoL'))
    lr = vec(rhoL)'*vec(rhoR)


    X = Matrix(cholesky(rhoR).U)
    Y = Matrix(cholesky(rhoL).U)
    F =svd(Y*Diagonal(l)*transpose(X))
    YU= inv(Y)*F.U
    VX= F.Vt*inv(transpose(X))
    @tensor A[:] := VX[-1,1]*A[1,-2,3]*YU[3,-3]
    l = F.S ./ LinearAlgebra.norm(F.S)
    d = size(A,2)

    return A,l
end

function canonicalize_one_site(A,l,Dmax=10,tol=1e-8,vR=[1],vL=[1])
    valR, rhoR = transfer_spectrum(A,l,vR,dir=:right)
    valL, rhoL = transfer_spectrum(A,l,vL,dir=:left)
    #println(valL[1]," :L _ R: ", valR[1])
    rhoR = canonicalize_eigenoperator(rhoR)
    rhoL = canonicalize_eigenoperator(rhoL)
    lr = vec(rhoL)'*vec(rhoR)

    evl, Ul = eigen(rhoL)
    evr, Ur = eigen(rhoR)
    sevr = sqrt.(complex.(evr))
    sevl = sqrt.(complex.(evl))
    X = Diagonal(sevr)[abs.(sevr) .> tol,:] * Ur'
    Y = Diagonal(sevl)[abs.(sevl) .> tol,:] * Ul'

    #X = Matrix(cholesky(rhoR).U)
    #Y = Matrix(cholesky(rhoL).U)

    F =svd(Y*Diagonal(l)*transpose(X))
    #U,S,Vt,D,err = truncate_svd(F)
    YU= pinv(Y)*F.U
    VX= F.Vt*pinv(transpose(X))
    @tensor A[:] := VX[-1,1]*A[1,-2,3]*YU[3,-3]
    #A = A ./ valL
    l = F.S ./ LinearAlgebra.norm(F.S)
    d = size(A,2)

    return A,l
end

"""
    transfer_spectrum(A,l,v;dir=:right)

get the spectrum of the transfer matrix with boundary conditions v in the space of dominant eigenvectors
"""
function transfer_spectrum(A,l,v; dir=:right)
    nev = length(v)
    vals, vecs = transfer_spectrum(A,l,dir=dir,nev=nev)
    if sum(vals .- vals[1]) != 0
        @warn "eigenvalues different"
        println(vals)
    end
    rho = reshape(vecs*vec(v),length(l),length(l))
    return vals[1], rho
end

function transfer_spectrum(A,B,lL,lM,lR;nev=1)
    TAR = transfer_right(lL,A)
    TBR = transfer_right(lM,B)
    TAL = transfer_left(A,lM)
    TBL = transfer_left(B,lR)
    TBAL = TBL*TAL
    TBAR = TAR*TBR
    if size(TBAL,1)<10
        vals, vecs = eigen(Matrix(TBAL))
        valR = vals[end-nev+1:end]
        rhoR = vecs[:,end-nev+1:end]
    else
        valR, rhoR = eigs(TBAL,nev=nev)
    end
    if size(TBAR,1)<10
        vals, vecs = eigen(Matrix(TBAR))
        valL = vals[end-nev+1:end]
        rhoL = vecs[:,end-nev+1:end]
    else
        valL, rhoL = eigs(TBAR,nev=nev)
    end
    return valR,rhoR,valL,rhoL
end


function canonicalize_two_site(A,B,lL,lM,lR,Dmax=10,tol=1e-8)
    TAR = transfer_right(lL,A)
    TBR = transfer_right(lM,B)
    TAL = transfer_left(A,lM)
    TBL = transfer_left(B,lR)
    #TABR = TBR*TAR
    #TABL = TAL*TBL
    TBAL = TBL*TAL
    TBAR = TAR*TBR
    if size(TBAL,1)<10
        vals, vecs = eigen(Matrix(TBAL))
        valR = vals[end]
        rhoR = vecs[:,end]
    else
        valR, rhoR = eigs(TBAL,nev=1)
    end
    if size(TBAR,1)<10
        vals, vecs = eigen(Matrix(TBAR))
        valL = vals[end]
        rhoL = vecs[:,end]
    else
        valL, rhoL = eigs(TBAR,nev=1)
    end
    #println(valL[1],"  ", valR[1])
    chi = Int(sqrt(length(rhoR)))
    rhoR = reshape(rhoR,chi,chi)
    phase = tr(rhoR)/abs(tr(rhoR))
    rhoR = rhoR ./ phase
    rhoR = Hermitian(0.5*(rhoR + rhoR'))
    chi = Int(sqrt(length(rhoL)))
    rhoL = reshape(rhoL,chi,chi)
    phase = tr(rhoL)/abs(tr(rhoL))
    rhoL = rhoL ./ phase
    rhoL = Hermitian(.5*(rhoL + rhoL'))


    lr = vec(rhoL)'*vec(rhoR)
    rhoL /= sqrt(abs(lr))
    rhoR /= sqrt(abs(lr))

    #Cholesky
    X = Matrix(cholesky(rhoR,check=false).U)
    Y = Matrix(cholesky(rhoL,check=false).U)
    F = svd(Y*Diagonal(lM)*transpose(X))
    #U,S,Vt,D,err = truncate_svd(F)


    #rest
    YU= inv(Y)*F.U #./ (valL[1])^(1/4)
    VX= F.Vt*inv(transpose(X)) #./ (valR[1])^(1/4)
    @tensor A[:] := A[-1,-2,3]*YU[3,-3]
    @tensor B[:] := VX[-1,1]*B[1,-2,-3]
    lM = F.S ./ LinearAlgebra.norm(F.S)
    d = size(A,2)
    B,lL,A = blockDecimateMPS(lM,lL,lM,B,A, reshape(Matrix(I,d^2,d^2),d,d,d,d),Dmax,tol)
    return A,B,lL,lM,lL
end

function hamblock_to_mpo(hamBlock, trunc::TruncationArgs)
    d = size(hamBlock,1)
    F = svd(reshape(permutedims(hamBlock,[1 3 2 4]),d^2,d^2))
    U,S,Vt,chi = truncate_svd(F, trunc)
    idMat = Matrix(1.0I,d,d)
    U = reshape(U*Diagonal(sqrt.(S)),d,d,chi)
    Vt = reshape(Diagonal(sqrt.(S))*Vt,chi,d,d)
    L = []
    R = []
    for k in 1:chi
        k==1 ? L = U[:,:,k] : L = [L; U[:,:,k]]
        k==1 ? R = Vt[k,:,:] : R = [R zeros(size(R,1),d); zeros(d,size(R,2)) Vt[k,:,:]]
    end
    hamMPO = permutedims(reshape([Matrix(1.0I,d,2d*(chi+1));
            L zeros(d*chi,d*(2chi+1));
            zeros(d*chi,d) R zeros(d*chi,d*(chi+1));
            zeros(d,d*(chi+1)) repeat(idMat,1,chi+1)],d,2+2chi,d,2+2chi),[2 1 3 4])
    return hamMPO
end

function ground_state(hamiltonian,trunc::TruncationArgs; test=false, verbose=false, nmax=1000)
    d = Int(sqrt(size(hamiltonian,1)))
    hamblock = reshape(hamiltonian,d,d,d,d)
    hamblock2 = reshape(hamiltonian*hamiltonian,d,d,d,d)
    hamMPO = hamblock_to_mpo(hamblock)

    idBlock=reshape(Matrix(I,d^2,d^2),d,d,d,d);
    A = B = reshape(Matrix(1.0I,d,1),1,d,1) #rand(Float64,5,d,5);
    lA = lB = [1]#repeat([1],5);

    dt = opnorm(hamiltonian)
    counter = 0
    # function test_gs()
    #     E = expectation_value_two_site(A,B,lA,lB,lA,ham2Block)
    #     if
    #         print("A")
    #     end
    #     return
    # end
    beta = 0
    energy = Float64[]
    Udt2 = reshape(exp(-dt*hamiltonian/2),d,d,d,d);
    Udt = reshape(exp(-dt*hamiltonian),d,d,d,d);
    mpsnorm = 0.5*(TN.expectation_value_two_site(B,A,lB,lA,lB,idBlock)+TN.expectation_value_two_site(A,B,lA,lB,lA,idBlock))
    c0 = counter
    while counter < 2 ||  counter < nmax #|| abs((energy[end]-energy[end-1])/energy[end]) > tol
        if mod(counter,50) == 0 || abs(mpsnorm-1) > min(dt,.1)
            if (verbose) println("canonicalize: ", mpsnorm) end
            A, B, lA, lB,lC = TN.canonicalize_two_site(A,B,lA,lB,lA,Dmax,tol)
            c0 = counter
        end
        #abs((energy[end]-energy[end-1])/energy[end])
        if counter - c0> 4 && abs(diff(energy)[end]) < dt^3  #&&mod(counter,50)==0
            print(counter, ", ")
            c0 = counter
            dt = 1/2*dt
            Udt2 = reshape(exp(-dt*hamiltonian/2),d,d,d,d);
            Udt = reshape(exp(-dt*hamiltonian),d,d,d,d);
            #println(abs((energy[end]-energy[end-1])/energy[end]))
            if (verbose) println("dt: ",dt) end
        end
        beta += dt
        A, lB, B, error = TN.blockDecimateMPS(lA,lB,lA,A,B,Udt2,Dmax,tol)
        B, lA, A, error = TN.blockDecimateMPS(lB,lA,lB,B,A,Udt,Dmax,tol)
        A, lB, B, error = TN.blockDecimateMPS(lA,lB,lA,A,B,Udt2,Dmax,tol)
        mpsnorm = 0.5*(TN.expectation_value_two_site(B,A,lB,lA,lB,idBlock)+TN.expectation_value_two_site(A,B,lA,lB,lA,idBlock))
        push!(energy,0.5*(TN.expectation_value_two_site(B,A,lB,lA,lB,hamblock)+TN.expectation_value_two_site(A,B,lA,lB,lA,hamblock)))
        counter+=1

        #err[n] += error
    end
    println("beta: ", beta)
    A, B, lA, lB,lC = TN.canonicalize_two_site(A,B,lA,lB,lA,Dmax,tol)
    return A,B,lA,lB, energy
end

""" Returns the second renyi entropy density:

        ```renyi2density_thermal(A,B,lA,lB) -> r2```"""
function renyi2density_thermal(A,B,lA,lB)
    B = deauxillerate_onesite(absorb_l(B,lA,:right))
    A = deauxillerate_onesite(absorb_l(A,lB,:right))
    sA=size(A)
    sB=size(B)
    function contract(R,A)
        #rhoR=Diagonal(conj(l))*rhoR*Diagonal(l)
        sA = size(A)
        temp = reshape(R,sA[4],sA[4],sA[4],sA[4])
        @tensoropt (r,-2,-3,-4) temp[:] := temp[r,-2,-3,-4]*conj(A[-1,-5,-6,r])
        @tensoropt (r,-2,-3,-4) temp[:] := temp[-1,r,-3,-4,c,-6]*A[-2,c,-5,r]
        @tensoropt (r,-2,-3,-4) temp[:] := temp[-1,-2,r,-4,c,-6]*conj(A[-3,-5,c,r])
        @tensoropt (r,-2,-3,-4) temp[:] := temp[-1,-2,-3,r,c,-6]*A[-4,c,-5,r]
        @tensoropt (-1,-2,-3,-4) temp[:] := temp[-1,-2,-3,-4,c,c]
        st = size(temp)
        return reshape(temp,st[1]*st[2]*st[3]*st[4])
    end
    contract2(R) = contract(contract(R,A),B)
    Tmap = LinearMap{ComplexF64}(contract2,sA[4]*sA[4]*sA[4]*sA[4])
    if sA[1] > 2
        vals, vecs = eigs(Tmap,nev = 1)
    else
        vals = eigen(Matrix(Tmap)).values
    end
    return -log(maximum(vals[1]))/2
end

function transfer_matrix_squared(g,l)
    A = deauxillerate_onesite(absorb_l(g,l,:right))
    sA=size(A)
    function contract(R)
        temp = reshape(R,sA[4],sA[4],sA[4],sA[4])
        @tensoropt (r,-2,-3,-4) temp[:] := temp[r,-2,-3,-4]*conj(A[-1,-5,-6,r])
        @tensoropt (r,-2,-3,-4) temp[:] := temp[-1,r,-3,-4,c,-6]*A[-2,c,-5,r]
        @tensoropt (r,-2,-3,-4) temp[:] := temp[-1,-2,r,-4,c,-6]*conj(A[-3,-5,c,r])
        @tensoropt (r,-2,-3,-4) temp[:] := temp[-1,-2,-3,r,c,-6]*A[-4,c,-5,r]
        @tensoropt (-1,-2,-3,-4) temp[:] := temp[-1,-2,-3,-4,c,c]
        st = size(temp)
        return reshape(temp,st[1]*st[2]*st[3]*st[4])
    end
    T = LinearMap{ComplexF64}(contract,sA[1]^4,sA[4]^4)
    return T
end

""" Returns the second renyi entropy for subsystems up to size n:

        ```renyi2_thermal(A,B,lA,lB,n) -> re```"""
function renyi2_thermal(A,B,lA,lB,n)
    B = deauxillerate_onesite(absorb_l(B,lA,:right))
    A = deauxillerate_onesite(absorb_l(A,lB,:right))
    re = Array{Float64,1}(undef,n)
    function funcB(rhoR)
        #rhoR=Diagonal(conj(l))*rhoR*Diagonal(l)
        @tensoropt (lu,ld,ru,rd,-1,-2,-3,-4) temp[:] := rhoR[lu,ld,-3,-5]*conj(B[-2,-6,a,ld])*B[-1,-4,a,lu]
        st = size(temp)
        return reshape(temp,st[1],st[2],st[3]*st[4],st[5]*st[6])
    end

    function funcA(rhoR)
        #rhoR=Diagonal(conj(l))*rhoR*Diagonal(l)
        @tensoropt (lu,ld,ru,rd,-1,-2,-3,-4) temp[:] :=  rhoR[lu,ld,-3,-5]*conj(A[-2,-6,a,ld])*A[-1,-4,a,lu]
        st = size(temp)
        return reshape(temp,st[1],st[2],st[3]*st[4],st[5]*st[6])
        return temp
    end
    @tensoropt (r1,r2,-4,-3) temp[:] := conj(B[-2,-4,a,r])*B[-1,-3,a,r]
    l=Matrix(Diagonal(lB.^2))
    @tensor temp2[:] := l[l1,l2]*temp[l1,l2,-1,-2]
    re[1] = real(tr(temp2*temp2))
    for k in 1:n-1
        if isodd(k)
            temp = funcA(temp)
            l=Matrix(Diagonal(lA.^2))
        else
            temp = funcB(temp)
            l=Matrix(Diagonal(lB.^2))
        end
        @tensor temp2[:] := l[l1,l2]*temp[l1,l2,-1,-2]
        re[k+1] = real(tr(temp2*temp2))
    end
    return re
end

""" Returns the entanglement entropy for subsystems up to size n:

        ```entanglement_entropy(A,B,lA,lB,n) -> ee```"""
function entanglement_entropy(A,B,lA,lB,n)
    B = absorb_l(B,lA,:right)
    A = absorb_l(A,lB,:right)
    ee = Array{Float64,1}(undef,n)
    function funcB(rhoR)
        #rhoR=Diagonal(conj(l))*rhoR*Diagonal(l)
        @tensoropt (r1,r2,-4,-3) temp[:] := (rhoR[r1,r2,-3,-5]*conj(B[-2,-6,r2]))*B[-1,-4,r1]
        st = size(temp)
        return reshape(temp,st[1],st[2],st[3]*st[4],st[5]*st[6])
    end
    function funcA(rhoR)
        #rhoR=Diagonal(conj(l))*rhoR*Diagonal(l)
        @tensoropt (r1,r2,-4,-3) temp[:] := (rhoR[r1,r2,-3,-5]*conj(A[-2,-6,r2]))*A[-1,-4,r1]
        st = size(temp)
        return reshape(temp,st[1],st[2],st[3]*st[4],st[5]*st[6])
    end

    @tensoropt (r1,r2,-4,-3) temp[:] := conj(B[-2,-4,r])*B[-1,-3,r]
    l=Matrix(Diagonal(lB.^2))
    @tensor temp2[:] := l[l1,l2]*temp[l1,l2,-1,-2]
    ee[1] = real(tr(-temp2*log(temp2)))

    for k in 1:n-1
        if isodd(k)
            temp = funcA(temp)
            l=Matrix(Diagonal(lA.^2))
        else
            temp = funcB(temp)
            l=Matrix(Diagonal(lB.^2))
        end
        @tensor temp2[:] := l[l1,l2]*temp[l1,l2,-1,-2]
        ee[k+1] = real(tr(-temp2*log(temp2)))
    end
    return ee
end


"""truncates the full MPS/MPO. There seems to be some bug """
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
        F = svd(tmp)
        U,S,V = truncate_svd(F,D)
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


"""
calculates Tr(mpo^n) for n=1,2,4
"""
function traceMPO(mpo,n=1)
    L = length(mpo)
    Array{ComplexF64,n}(undef,ntuple(i->1,n)...)
    F[1] = 1
    if n == 1
        for i = 1:L
            @tensor F[-2] := F[1]*mpo[i][1,2,2,-2]
        end
        return F[1]
    elseif n == 2
        for i = 1:L
            @tensor F[-3,-4] := F[1,2]*mpo[i][1,3,4,-3]*conj(mpo[i][2,3,4,-4])
        end
        return F[1,1]
    elseif n == 4
        for i = 1:L
            @tensor F[-5,-6,-7,-8] := F[1,2,3,4]*mpo[i][1,5,6,-5]*conj(mpo[i][2,7,6,-6])*conj(mpo[i][3,8,7,-7])*mpo[i][4,8,5,-8]
        end
        return F[1,1,1,1]
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
        F = Array{ComplexF64,2}(undef,1,1)
        F[1,1] = 1
        for i = 1:L
            @tensor F[-3,-4] := F[1,2]*mpo1[i][1,3,4,-3]*conj(mpo2[i][2,3,4,-4])
        end
        return F[1,1]
    elseif n == 2
        F = Array{Complex64,3}(undef,1,1,1)
        F[1,1,1] = 1
        for i = 1:L
            @tensor F[-4,-5,-6] := F[1,2,3]*mpo1[i][1,4,5,-4]*conj(mpo1[i][2,6,5,-5])*mpo2[i][3,6,4,-6]
        end
        return F[1,1,1]
    else
        println("ERROR: choose n=1,2 in traceMPOprod(mpo1,mpo2,n=1)")
        return "nan"
    end
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
