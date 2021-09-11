using Test, TensorNetworks, TensorOperations, LinearAlgebra

@testset "Gate" begin
    op = rand(2,2)
    H = op+op'
    U = exp(H*1.0im)
    @test ishermitian(op) == ishermitian(Gate(op))
    @test ishermitian(Gate(H))
    @test isunitary(op) == isunitary(Gate(op))
    @test isunitary(Gate(U))
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
    states, energies = eigenstates(ham, mps, 5; precision = 1e-8)
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