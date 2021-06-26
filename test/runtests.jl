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
