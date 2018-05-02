module TEBD
using TensorOperations
using MPS

# define Pauli matrices
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]

function TwoSiteIsingHamiltonian(J,h,g)
    ZZ = kron(sz, sz)
    ZI = kron(sz, si)
    IZ = kron(si, sz)
    XI = kron(sx, si)
    IX = kron(si, sx)
    H = J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ)
    return H
end

function TwoSiteHeisenbergHamiltonian(Jx,Jy,Jz,hx)
    XX = kron(sx, sx)
    YY = kron(sy, sy)
    ZZ = kron(sz, sz)
    XI = kron(sx, si)
    IX = kron(si, sx)
    H = Jx*XX + Jy*YY + Jz*ZZ + hx/2*(XI+IX)
    return H
end

function truncate_svd(U, S, V, D)
    U = U[:, 1:D]
    S = S[1:D]
    V = V[1:D, :]
    return U, S, V
end

""" block_decimation(W, Tl, Tr, Dmax)
Apply two-site operator W (4 indexes) to mps tensors Tl (left) and Tr (right)
and performs a block decimation (one TEBD step)
```block_decimation(W,TL,TR,Dmax) -> Tl, Tr"""
function block_decimation(W, Tl, Tr, Dmax)
    ### input:
    ###     W:      time evolution op W=exp(-tau h) of size (d,d,d,d)
    ###     Tl, Tr: mps sites mps[i] and mps[i+1] of size (D1l,d,D1r) and (D2l,d,D2r)
    ###     Dmax:   maximal bond dimension
    ### output:
    ###     Tl, Tr after one time evolution step specified by W

    D1l,d,D1r = size(Tl)
    D2l,d,D2r = size(Tr)

    # absorb time evolution gate W into Tl and Tr
    @tensor theta[-1,-2,-3,-4] := Tl[-1,2,3]*W[2,4,-2,-3]*Tr[3,4,-4] # = (D1l,d,d,D2r)
    theta = reshape(theta, D1l*d,d*D2r)
    U,S,V = svd(theta, thin=true)
    V = V'
    D1 = size(S)[1] # number of singular values

    if D1 <= Dmax
        Tl = reshape(U*diagm(sqrt.(S)), D1l,d,D1)
        Tr = reshape(diagm(sqrt.(S))*V, D1,d,D2r)
    else
        U,S,V = truncate_svd(U,S,V,Dmax)
        Tl = reshape(U*diagm(sqrt.(S)), D1l,d,Dmax)
        Tr = reshape(diagm(sqrt.(S))*V, Dmax,d,D2r)
    end

    return Tl, Tr
end

function time_evolve(mps, block, total_time, steps, D, mpo=nothing)
    ### block = hamiltonian
    ### use negative imaginary total_time for imaginary time evolution

    d = size(mps[1])[2]
    L = length(mps)
    W = expm(-1im*total_time*block/steps)
    W = reshape(W, (d,d,d,d))
    expect = Array{Any}(steps,2)

    for counter = 1:steps
        for i = 1:2:L-1 # odd sites
            mps[i], mps[i+1] = block_decimation(W, mps[i], mps[i+1], D)
        end

        for i = 2:2:L-1 # even sites
            mps[i], mps[i+1] = block_decimation(W, mps[i], mps[i+1], D)
        end
        if imag(total_time) != 0.0 # normalization in case of imaginary time evolution
            MPS.makeCanonical(mps)
        end

        ## expectation values:
        if mpo != nothing
            expect[counter,:] = [counter*total_time/steps MPS.mpoExpectation(mps,mpo)]
        end
    end

    return expect
end

function time_evolve_mpoham(mps, block, total_time, steps, D, mpo=nothing)
    ### block = hamiltonian
    ### use negative imaginary total_time for imaginary time evolution
    stepsize = total_time/steps
    d = size(mps[1])[2]
    L = length(mps)

    expect = Array{Any}(steps,2)

    for counter = 1:steps
        for i = 1:2:L-1 # odd sites
            W = expm(-1im*stepsize*block(i,counter*total_time/steps))
            W = reshape(W, (d,d,d,d))
            mps[i], mps[i+1] = block_decimation(W, mps[i], mps[i+1], D)
        end

        for i = 2:2:L-1 # even sites
            W = expm(-1im*stepsize*block(i,counter*total_time/steps))
            W = reshape(W, (d,d,d,d))
            mps[i], mps[i+1] = block_decimation(W, mps[i], mps[i+1], D)
        end
        if imag(total_time) != 0.0 # normalization in case of imaginary time evolution
            MPS.makeCanonical(mps)
        end

        ## expectation values:
        if mpo != nothing
            expect[counter,:] = [counter*total_time/steps MPS.mpoExpectation(mps,mpo)]
        end
    end

    return expect
end


end
