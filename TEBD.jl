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


""" returns the Ising parameters after every time evolution step """
function evolveIsingParams(J0, h0, g0, time)
    ### time evolution of all quench parameters
    J = J0
    h = h0 #+ exp(-3(time-2)^2)
    g = g0

    return J, h, g
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

""" returns the Heisenberg parameters after every time evolution step """
function evolveHeisenbergParams(Jx0, Jy0, Jz0, hx0, time)
    ### time evolution of all quench parameters
    Jx = Jx0
    Jy = Jy0
    Jz = Jz0
    hx = hx0 #+ exp(-(time-2)^2)

    return Jx, Jy, Jz, hx
end


function truncate_svd(U, S, V, D)
    U = U[:, 1:D]
    S = S[1:D]
    V = V[1:D, :]
    return U, S, V
end


""" block_decimation(W, Tl, Tr, Dmax,dir)
Apply two-site operator W (4 indexes) to mps tensors Tl (left) and Tr (right)
and performs a block decimation (one TEBD step)
```block_decimation(W,TL,TR,Dmax,dir) -> Tl, Tr"""
function block_decimation(W, Tl, Tr, Dmax, dir)
    ### input:
    ###     W:      time evolution op W=exp(-tau h) of size (d,d,d,d)
    ###     Tl, Tr: mps sites mps[i] and mps[i+1] of size (D1l,d,D1r) and (D2l,d,D2r)
    ###     Dmax:   maximal bond dimension
    ###     dir:    direction (-1 is leftcanonical, +1 is rightcanonical) for preference where to put singular value matrix during sweep
    ### output:
    ###     Tl, Tr after one time evolution step specified by W

    stl = size(Tl)
    str = size(Tr)
    if length(stl)==4
        Tl = reshape(permutedims(Tl, [1,3,2,4]), stl[1]*stl[3],stl[2],stl[4])
        Tr = reshape(Tr, str[1],str[2],str[3]*str[4])
    end
    D1l,d,D1r = size(Tl)
    D2l,d,D2r = size(Tr)

    # absorb time evolution gate W into Tl and Tr
    @tensor theta[-1,-2,-3,-4] := Tl[-1,2,3]*W[2,4,-2,-3]*Tr[3,4,-4] # = (D1l,d,d,D2r)
    # println("theta", typeof(theta))
    theta = reshape(theta, D1l*d,d*D2r)
    U,S,V = svd(theta, thin=true)
    # println(typeof(theta))
    # SVD = svds(theta, nsv=min(Dmax,D1l*d,d*D2r))[1]
    # println("svds")
    # U = SVD[:U]
    # S = SVD[:S]
    # V = SVD[:Vt]
    # println("svd")
    V = V'
    D1 = size(S)[1] # number of singular values

    if D1 <= Dmax
        if dir == -1
            Tl = reshape(U, D1l,d,D1)
            Tr = reshape(diagm(S)*V, D1,d,D2r)
        else
            Tl = reshape(U*diagm(S), D1l,d,D1)
            Tr = reshape(V, D1,d,D2r)
        end
    else
        U,S,V = truncate_svd(U,S,V,Dmax)
        if dir == -1
            Tl = reshape(U, D1l,d,Dmax)
            Tr = reshape(diagm(S)*V, Dmax,d,D2r)
        else
            Tl = reshape(U*diagm(S), D1l,d,Dmax)
            Tr = reshape(V, Dmax,d,D2r)
        end
    end
    # println("trunc")

    if length(stl)==4
        Tl = permutedims(reshape(Tl, stl[1],stl[3],stl[2],min(D1,Dmax)), [1,3,2,4])
        Tr = reshape(Tr, min(D1,Dmax),str[2],str[3],str[4])
    end

    return Tl, Tr
end


function time_evolve_mpoham(mps, block, total_time, steps, D, entropy_cut, params, eth, mpo=nothing)
    ### block = hamiltonian
    ### use -im*total_time for imaginary time evolution
    ### assumption: start with rightcanonical mps
	### eth = (true,E1,hamiltonian) --> do ETH calcs if true for excited energy E1 wrt hamiltonian
    stepsize = total_time/steps
    d = size(mps[1])[2]
    L = length(mps)
    if isodd(L)
        even_start = L-1
        odd_length = true
    else
        even_start = L-2
        odd_length = false
    end
    mpo_to_mps_trafo = false
    if length(size(mps[1])) == 4 # control variable to make mps out of mpo
        mpo_to_mps_trafo = true
    end
    if !mpo_to_mps_trafo && MPS.check_LRcanonical(mps[1],-1) # use rightcanonical mps as default input for sweeping direction
        MPS.makeCanonical(mps)
    end

    expect = Array{Any}(steps,2)
    entropy = Array{Any}(steps,2)

    for counter = 1:steps
        time = counter*total_time/steps

        ## ************ right sweep over odd sites
        for i = 1:2:L-1
            W = expm(-1im*stepsize*block(i,time,params))
            W = reshape(W, (d,d,d,d))
            mps[i], mps[i+1] = block_decimation(W, mps[i], mps[i+1], D, -1)
			# preserve canonical structure:
            if mpo_to_mps_trafo # brute force, but works easily for mpo
                MPS.makeCanonical(mps,i+2)
            else # more efficiently for pure mps
                mps[i+1],R,DB = MPS.LRcanonical(mps[i+1],-1) # leftcanonicalize current sites
                if i < L-1 || odd_length
                    @tensor mps[i+2][-1,-2,-3] := R[-1,1]*mps[i+2][1,-2,-3]
                end
            end
        end

        ## ************ left sweep over even sites
        if !mpo_to_mps_trafo
            mps[L],R,DB = MPS.LRcanonical(mps[L],1) # rightcanonicalize at right end
            @tensor mps[L-1][:] := mps[L-1][-1,-2,1]*R[1,-3]
        end
        for i = even_start:-2:2
            W = expm(-1im*stepsize*block(i,time,params))
            W = reshape(W, (d,d,d,d))
            mps[i], mps[i+1] = block_decimation(W, mps[i], mps[i+1], D, 1)
			# preserve canonical structure:
            if mpo_to_mps_trafo
                MPS.makeCanonical(mps,i-2)
            else
                mps[i],R,DB = MPS.LRcanonical(mps[i],1) # rightcanonicalize current sites
                @tensor mps[i-1][:] := mps[i-1][-1,-2,1]*R[1,-3]
            end
        end
        if !mpo_to_mps_trafo
            mps[1],R,DB = MPS.LRcanonical(mps[1],1) # rightcanonicalize at left end
        end

        ## expectation values:
        if mpo != nothing
            if mpo == "Ising"
                J0, h0, g0 = params
                J, h, g = evolveIsingParams(J0, h0, g0, time)
                hamiltonian = MPS.IsingMPO(L, J, h, g)
                expect[counter,:] = [time MPS.mpoExpectation(mps,hamiltonian)]
            elseif mpo == "Heisenberg"
                Jx0, Jy0, Jz0, hx0 = params
                Jx, Jy, Jz, hx = evolveHeisenbergParams(Jx0, Jy0, Jz0, hx0, time)
                hamiltonian = MPS.HeisenbergMPO(L, Jx, Jy, Jz, hx)
                expect[counter,:] = [time MPS.mpoExpectation(mps,hamiltonian)]
            else
                expect[counter,:] = [time MPS.mpoExpectation(mps,mpo)]
            end
        end

        ## entanglement entropy:
        if entropy_cut > 0
            entropy[counter,:] = [time MPS.entropy(mps,entropy_cut)]
        end

		## ETH calculations:
		if eth[1] == true
			E1, hamiltonian = real(eth[2]), eth[3]
			rho = MPS.multiplyMPOs(mps,mps)
			E_thermal = real(MPS.traceMPO(MPS.multiplyMPOs(rho,hamiltonian)))
			if E_thermal <= E1
				return E_thermal, real(time*1im) # im*time = beta/2
			end
		end
    end

    return expect, entropy
end


end
