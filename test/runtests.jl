using Test, TensorNetworks, TensorOperations, LinearAlgebra

@testset "basic operations" begin
    M = rand(ComplexF64,5,2,5)
    ML, RL, DB = TensorNetworks.LRcanonical(M, :left)
    @tensor cL[:] := conj(ML[1,2,-1])*ML[1,2,-2]
    @test cL ≈ Matrix(1.0I, 5,5)

    MR, RR, DB = TensorNetworks.LRcanonical(M, :right)
    @tensor cR[:] := conj(MR[-1,2,1])*MR[-2,2,1]
    @test cR ≈ Matrix(1.0I,5,5)
    #trunc = TruncationArgs()
    #truncate_svd(M, )
end

@testset "Imaginary TEBD" begin
    #Ground state energy of Ising CFT
    Nchain = 20
    Dmax = 10
    ham = isingHamGates(Nchain,1,1,0)
    hamMPO = IsingMPO(Nchain,1,1,0)
    mps = canonicalize(identityOpenMPS(ComplexF64, Nchain, 2, truncation = TruncationArgs(Dmax, 1e-12, true)))
    states = get_thermal_states(mps, ham, 30, .1, order=2)
    energy = expectation_value(states[1][1],hamMPO)
    @test abs(energy/(Nchain-1) + 4/π) < 1/Nchain

    ham = isingHamGates(Nchain,1,1,0)[2:3]
    mps = canonicalize(identityUMPS(ComplexF64, 2, 2, truncation = TruncationArgs(Dmax, 1e-12, true)))
    states = get_thermal_states(mps, ham, 30, .1, order=2)
    energy = (expectation_value(states[1][1],ham[1],1) + expectation_value(states[1][1],ham[2],2))/2
    @test abs(energy + 4/π) < 1/Nchain
end

@testset "DMRG" begin
    #Test a few low lying eigenstates of a simple Ising
    Nchain = 5
    Dmax=20
    ham = IsingMPO(Nchain, 1, 0, 0)
    mps = canonicalize(randomOpenMPS(ComplexF64, Nchain, 2, Dmax, purification = false))
    states, energies = eigenstates(mps, ham, 1e-10, 5)
    @test energies ≈ -[Nchain-1, Nchain-1, Nchain-3, Nchain-3, Nchain-3]

    #Ground state energy of Ising CFT
    Nchain = 20
    Dmax = 30
    ham = IsingMPO(Nchain, 1, 1, 0)
    mps = canonicalize(randomOpenMPS(ComplexF64, Nchain, 2, Dmax, purification = false))
    states, energies = eigenstates(mps, ham, 1e-8, 1)
    @test abs(energies[1]/(Nchain-1) + 4/π) < 1/Nchain
end
