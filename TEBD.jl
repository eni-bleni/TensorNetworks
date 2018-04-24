module TEBD
using TensorOperations



# define Pauli matrices
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]

function IsingHamiltonian(J,h,g)
    ZZ = kron(sz, sz)
    ZI = kron(sz, si)
    IZ = kron(si, sz)
    XI = kron(sx, si)
    IX = kron(si, sx)
    H = J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ)
    return H
end


function truncate_svd(U, S, V, D)
    U = U[:, 1:D]
    S = S[1:D]
    V = V[1:D, :]
    return U, S, V
end


""" block_decimation(W, Tl, Tr, Dmax)
employs the time evolution operator W=exp(-tau h) to mps tensors Tl (left) and Tr (right)
and performs a block decimation (one TEBD step)
"""
function block_decimation(W, Tl, Tr, Dmax)
    ### input:
    ###     W:      time evolution op W=exp(-tau h) of size (d,d,d,d)
    ###     Tl, Tr: mps sites mps[i] and mps[i+1] of size (D1l,d,D1r) and (D2l,d,D2r)
    ###     Dmax:   maximal bond dimension
    ### output:
    ###     Tl, Tr after one time evolution step specified by W

    D1l,d,D1r = size(Tl)
    D2l,d,D2r = size(Tr)
    @tensor begin # absorb time evolution gate W into Tl and Tr
        theta[-1,-2,-3,-4] := Tl[-1,2,3]*W[2,4,-2,-3]*Tr[3,4,-4] # = (D1l,d,d,D2r)
    end
    theta = reshape(theta, D1l*d,d*D2r)
    U,S,V = svd(theta, thin=true)
    V = V'
    D1 = size(S)[1] # number of singular values

    if D1 <= Dmax
        Tl = reshape(U, D1l,d,D1)
        Tr = reshape(diagm(S)*V, D1,d,D2r)
    else
        U,S,V = truncate_svd(U,S,V,Dmax)
        Tl = reshape(U, D1l,d,Dmax)
        Tr = reshape(diagm(S)*V, Dmax,d,D2r)
    end

    return Tl, Tr
end



# ## method 1: (as in Matlab code)
# h = TEBD.IsingHamiltonian(1,0,0)
# w = expm(-dt*h)
# U,S,V = svd(w)
# S = diagm(S)
# V = V' # only as shorthand notation:  w == svd(w) == U*S*V
# eta = size(S,1)
# println("eta = ", eta)
# U = U*sqrt(S)
# V = sqrt(S)*V

# MPO_even = MPO_odd = Array{Any}(N)
# si = [1 0; 0 1]
# for i = 1:N
#     MPO_even[i] = MPO_odd[i] = si
# end
# for i = 1:2:N-1
#     MPO_odd[i] = U
#     MPO_odd[i+1] = V
# end
# for i = 2:2:N-1
#     MPO_even[i] = U
#     MPO_even[i+1] = V
# end


end
