using Test, TensorNetworks, TensorOperations, LinearAlgebra

@testset "Gate" begin
    op = rand(Float64,2,2)
    g = Gate(op)
    for k1 in 1:2
        for k2 in 1:2
        @test g[k1,k2] == op[k1,k2]
        end
    end

    @test typeof(g) == GenericSquareGate{Float64,2}
    @test complex(typeof(g)) == GenericSquareGate{ComplexF64,2}

    H = (op+op')/2
    U = exp(H*1.0im)
    @test ishermitian(op) == ishermitian(g)
    @test ishermitian(Gate(Matrix(H)))
    @test isunitary(op) == isunitary(g)
    @test isunitary(Gate(U))

    id = IdentityGate;
    z = rand(ComplexF64)
    @test data(z*id) == z
    @test data(id*z) == z
    @test data(z*g) ≈ z*data(g)
    @test data(g*z) ≈ z*data(g)

    @test g+g ≈ 2*g ≈ data(g)+data(g) 
    @test id+id ≈ 2*id 
    @test g + z*id ≈ data(g) + z*one(data(g))
    @test z*id + g  ≈ data(g) + z*one(data(g))

    @test data(exp(z*id)) ≈ exp(z) 
    @test data(exp(g)) ≈ exp(data(g)) 
    
    @test id' == id
    @test data((z*id)') ≈ z'

    @test Hermitian(z*g) ≈ Gate((z*op)'+(z*op))/2

    site = qubit(rand(),rand())
    expval = vec(data(site))'*op*vec(data(site))
    @test expectation_value([site],g) ≈ expval
    g2 = Gate(TensorNetworks.gate(kron(op,op),2));
    @test expectation_value([site,site],g2) ≈ expval^2

    @test expectation_value([site], z*id) ≈ z
    @test expectation_value([site,site], z*id) ≈ z
    @test expectation_value([site,site,site], z*id) ≈ z

    D = 10;
    d = 2;
    site = randomGenericSite(D,d,D);

    d = 4;
    mat = rand(ComplexF64,d,d);
    gate = Gate(mat);
    @test data(gate) ≈ mat
    @test data(gate') ≈ mat'
    @test data(gate*gate) ≈ mat*mat


    tens = rand(ComplexF64,2,2,2,2);
    gate = Gate(tens);
    @test data(gate) ≈ tens
    @tensor tensc[:] := conj(tens[-3,-4,-1,-2]);
    @test data(gate') ≈ tensc
    @tensor t2[:] := tens[-1,-2,1,2]*tens[1,2,-3,-4];
    @test data(gate*gate) ≈ t2

    mat = rand(ComplexF64,d,d);
    mataux = kron(Matrix{ComplexF64}(I,d,d),mat);
    gate = Gate(mat);
    gateaux = TensorNetworks.auxillerate(gate);
    @test data(gateaux) ≈ mataux

    @tensor tens[:] := mat[-1,-3]*mat[-2,-4];
    @tensor tensaux[:] := mataux[-1,-3] *mataux[-2,-4];
    gateaux = TensorNetworks.auxillerate(Gate(tens));
    @test data(gateaux) ≈ tensaux


end

@testset "Canonicalize" begin
    N = 10
    c = 5
    Γ = randomOpenMPS(N,2,5).Γ

    L,r = TensorNetworks.to_left_orthogonal(Γ[1], method=:qr)
    @test isleftcanonical(L)
    @test TensorNetworks.data(Γ[1]) ≈ TensorNetworks.data(L*r)
    L,r = TensorNetworks.to_left_orthogonal(Γ[1], method=:svd)
    @test isleftcanonical(L)

    R,l = TensorNetworks.to_right_orthogonal(Γ[1], method=:qr)
    @test isrightcanonical(R)
    @test TensorNetworks.data(Γ[1]) ≈ TensorNetworks.data(l*R)
    R,l = TensorNetworks.to_right_orthogonal(Γ[1], method=:svd)
    @test isrightcanonical(R)

    Γc = TensorNetworks.to_left_right_orthogonal(Γ,center = 5)
    for k in 1:c-1
        @test isleftcanonical(Γc[k])
    end 
    @test norm(Γc[c]) ≈ 1
    for k in c+1:N
        @test isrightcanonical(Γc[k])
    end
end 

@testset "LCROpenMPS" begin
    N = 5
    mps = randomLCROpenMPS(N,2,5)
    for n in 0:N+1
        mps = canonicalize(mps,center = n)
        for k in 1:n-1
            @test isleftcanonical(mps[k])
        end
        if 0<n<N+1
            @test norm(mps[n]) ≈ 1
        end
        for k in n+1:N
            @test isrightcanonical(mps[k])
        end
    end
    for n in 0:N+1
        set_center!(mps,n)
        for k in 1:n-1
            @test isleftcanonical(mps[k])
        end
        if 0<n<N+1
            @test norm(mps[n]) ≈ 1
        end
        for k in n+1:N
            @test isrightcanonical(mps[k])
        end
    end

    mps = randomLCROpenMPS(N,2,5, purification=true)
    for n in 0:N+1
        mps = canonicalize(mps, center = n)
        for k in 1:n-1
            @test isleftcanonical(mps[k])
        end
        if 0<n<N+1
            @test norm(mps[n]) ≈ 1
        end
        for k in n+1:N
            @test isrightcanonical(mps[k])
        end
    end
end

@testset "Conversion" begin
    N = 10
    mps = canonicalize(randomLCROpenMPS(N,2,5))
    @test scalar_product(mps,mps) ≈ 1
    mps2 = OpenMPS(mps)
    @test scalar_product(mps,mps2) ≈ 1
    mps3 = LCROpenMPS(mps2)
    @test scalar_product(mps2,mps3) ≈ 1
    @test scalar_product(mps,mps3) ≈ 1
end

@testset "Transfer" begin
    D = 10;
    d = 2;
    site = randomGenericSite(D,d,D);
    R = randomRightOrthogonalSite(D,d,D);
    L = randomLeftOrthogonalSite(D,d,D);
    LR = randomOrthogonalLinkSite(D,d,D);
    id = Matrix{ComplexF64}(I,D,D);
    idvec = vec(id);
    T = transfer_matrix(site,:left)
    @test size(T) == (D^2,D^2)
    @test Matrix(T') ≈ Matrix(T)'
    @test transpose(Matrix(T)) ≈ Matrix(transfer_matrix(site,:right))

    z = rand(ComplexF64)
    @test Matrix(T) ≈ Matrix(transfer_matrix(site,IdentityGate))
    @test z*Matrix(T) ≈ Matrix(transfer_matrix(site,z*IdentityGate))

    @test idvec ≈ transfer_matrix(R)*idvec
    @test idvec ≈ transfer_matrix(L,:right)*idvec
    @test idvec ≈ transfer_matrix(LR,:right)*idvec
    @test idvec ≈ transfer_matrix(LR,:left)*idvec
    
    T1 = transfer_matrix(site,sz);
    @test size(T1) == (D^2,D^2)
    @test Matrix(T1') ≈ Matrix(T1)'

    g2 = Gate(TensorNetworks.gate(kron(sz,sz),2));
    T2 = transfer_matrix([site,site], g2);
    @test Matrix(T2) ≈ Matrix(T1*T1)
    @test transpose(Matrix(T2)) ≈ Matrix(transfer_matrix([site,site], g2,:right))

    g3 = Gate(TensorNetworks.gate(kron(sz,sz,sz),3));
    T3 = transfer_matrix([site,site,site], g3);
    @test Matrix(T3) ≈ Matrix(T1*T1*T1)
    @test transpose(Matrix(T3)) ≈ Matrix(transfer_matrix([site,site,site], g3,:right))

    g4 = Gate(TensorNetworks.gate(kron(sz,sz,sz,sz),4));
    T4 = transfer_matrix([site,site,site,site], g4);
    @test Matrix(T4) ≈ Matrix(T1*T1*T1*T1)
    @test transpose(Matrix(T4)) ≈ Matrix(transfer_matrix([site,site,site,site], g4,:right))
end

@testset "Compression" begin
    mps = canonicalize(randomOpenMPS(7,2,5));
    ΓL = mps[3];
    ΓR = mps[4];
    ΓL2, ΓR2, err = compress(ΓL, ΓR, mps.truncation);
    T = transfer_matrix(ΓL,:left) * transfer_matrix(ΓR,:left)
    T2 =transfer_matrix(ΓL2,:left) * transfer_matrix(ΓR2,:left)
    @test ΓL.Λ1 ≈ ΓL2.Λ1 && ΓL.Λ2 ≈ ΓL2.Λ2 && ΓR.Λ2 ≈ ΓR2.Λ2 && Matrix(T2) ≈ Matrix(T)

    thetaL,thetaR,phiL,phiR = 2*pi*rand(4);
    ΓL = qubit(thetaL,phiL);
    ΓR = qubit(thetaR,phiR);
    @test norm(ΓL) ≈ norm(ΓR) ≈ 1

    gL = exp(1im*rand(4)'*[si, sx, sy, sz]);
    gR = exp(1im*rand(4)'*[si, sx, sy, sz]);
    L = gL*vec(data(ΓL));
    R = gR*vec(data(ΓR));

    g = Gate(TensorNetworks.gate(kron(gR,gL),2));
    ΓL, S, ΓR, err = apply_two_site_gate(ΓL,ΓR,g, mps.truncation);
    @test err < 1e-16
    @test data(S) ≈ [1]
    #@test vec(data(ΓL)) ≈ L && vec(data(ΓR)) ≈ R
    @tensor ΓLR[:] := data(ΓL)[-1,-2,1] * data(ΓR)[1,-3,-4];
    @tensor LR[:] := reshape(L,1,2,1)[-1,-2,1] * reshape(R,1,2,1)[1,-3,-4];
    @test ΓLR ≈ LR

end


@testset "TEBD" begin
    d=4
    mat = rand(ComplexF64,d,d)
    g = Gate(mat)
    expmat = exp(1im*mat)
    id = one(mat)

    layers = TensorNetworks.st1gates(0,[g]);
    [@test data(l[1]) ≈ id for l in layers]
    layers = TensorNetworks.st1gates(1,[g]);
    @test data(layers[1][1]) ≈ expmat

    layers = TensorNetworks.st2gates(0,[g]);
    [@test data(l[1]) ≈ id for l in layers]
    layers = TensorNetworks.st2gates(1,[g]);
    @test data(*([l[1] for l in layers[1:2:end]]...)) ≈ expmat

    layers = TensorNetworks.frgates(0,[g]);
    [@test data(l[1]) ≈ id for l in layers]
    layers = TensorNetworks.frgates(1,[g]);
    @test data(*([l[1] for l in layers[1:2:end]]...)) ≈ expmat
end

@testset "Imaginary TEBD" begin
    #Ground state energy of Ising CFT
    Nchain = 20
    Dmax = 10
    ham = isingHamGates(Nchain,1,1,0)
    hamMPO = IsingMPO(Nchain,1,1,0)
    mps = canonicalize(identityOpenMPS(Nchain, 2, truncation = TruncationArgs(Dmax, 1e-12, true)))
    states, betas = get_thermal_states(mps, ham, 30, .1, order=2)
    energy = expectation_value(states[1],hamMPO)
    @test abs(energy/(Nchain-1) + 4/π) < 1/Nchain

    ham = isingHamGates(Nchain,1,1,0)[2:3]
    mps = canonicalize(identityUMPS(2, 2, truncation = TruncationArgs(Dmax, 1e-12, true)))
    states, betas = get_thermal_states(mps, ham, 30, .1, order=2)
    energy = (expectation_value(states[1],ham[1],1) + expectation_value(states[1],ham[2],2))/2
    @test abs(energy + 4/π) < 1/Nchain
end

@testset "DMRG" begin
    #Test a few low lying eigenstates of a simple Ising
    Nchain = 5
    Dmax = 10
    ham = IsingMPO(Nchain, 1, 0, 0)
    mps = canonicalize(randomLCROpenMPS(Nchain, 2, Dmax))
    states, energies = eigenstates(ham, mps, 5; precision = 1e-8)
    @test sort(energies) ≈ -[Nchain-1, Nchain-1, Nchain-3, Nchain-3, Nchain-3]

    #Ground state energy of Ising CFT
    Nchain = 20
    Dmax = 20
    ham = IsingMPO(Nchain, 1, 1, 0)
    mps = canonicalize(randomLCROpenMPS(Nchain, 2, Dmax))
    states, energies = eigenstates(ham, mps, 5; precision = 1e-8,alpha=1.0)
    @test abs(energies[1]/(Nchain-1) + 4/π) < 1/Nchain
end

@testset "UMPS expectation values" begin
    theta = 2*pi*rand()
    phi = 2*pi*rand()
    h = rand()
    g = rand()
    mps = TensorNetworks.productUMPS(theta,phi)
    hammpo = MPO(IsingMPO(5,1,h,g)[3]);
    hamgates = isingHamGates(5,1,h,g)[2:3];
    E = expectation_value(mps,hamgates[1],1)
    e0, heff,info = TensorNetworks.effective_hamiltonian(mps,hammpo,direction=:left);
    Eanalytic = - (cos(2*theta)^2 + h*sin(2*theta)*cos(phi) + g*cos(2*theta))
    #Empo = expectation_value(mps,hammpo)
    @test E ≈ e0 ≈ Eanalytic

    mps = randomUMPS(ComplexF64,1,2,1)
    canonicalize!(mps)
    E = expectation_value(mps,hamgates[1],1)
    e0, heff,info = TensorNetworks.effective_hamiltonian(mps,hammpo,direction=:left);
    #Empo = expectation_value(mps,hammpo)
    @test E ≈ e0
end