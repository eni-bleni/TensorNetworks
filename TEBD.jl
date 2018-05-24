# using MPSModule
using TensorOperations
# include("mpostruct.jl")
# using MPS
# define Pauli matrices
sx = [0 1; 1 0]
sy = [0 1im; -1im 0]
sz = [1 0; 0 -1]
si = [1 0; 0 1]
s0 = [0 0; 0 0]

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


struct quench
    hamblock
    hamMPO
    uMPO
    operators
end

function trotterblocks_timestep_mpo(block,L,dt,time)
    function w(i,L)
        b = expm(-1im*dt*block(i,L,time))
        s=size(b)
        return b
    end
    return blocks_to_mpo(w,L)
end

function blocks_to_mpo(block,L,D=Inf)
    mpo = Array{Array{Complex128,4}}(L)
    b = block(1,L)
    d = Int(sqrt(size(b)[1]))
    b = reshape(b,d,d,d,d)
    UL,VL = split(permutedims(b,[1 3 4 2]),D)
    @tensor mpo[1][-1,-2,-3,-4] := ones(1)[-1]*UL[-3,-2,-4]
    for i=2:L-1
        b = block(i,L)
        s = Int(sqrt(size(b)[1]))
        b = reshape(b,d,d,d,d)
        UR,VR = split(permutedims(b,[1 3 4 2]),D)
        if iseven(i)
            @tensor mpo[i][-1,-2,-3,-4] := VL[-1,1,-3]*UR[1,-2,-4]
        else
            @tensor mpo[i][-1,-2,-3,-4] := UR[-3,1,-4]*VL[-1,-2,1]
        end
        UL=UR
        VL=VR
    end
    @tensor mpo[L][-1,-2,-3,-4] := VL[-1,-2,-3]*ones(1)[-4]
    return MPO(mpo)
end

function time_evolve_simpler(mps, quench, total_time, steps, D)
    ### block = hamiltonian
    ### use -im*total_time for imaginary time evolution
    ### assumption: start with rightcanonical mps
	### eth = (true,E1,hamiltonian) --> do ETH calcs if true for excited energy E1 wrt hamiltonian
    dt = total_time/steps
    L = length(mps)
    nops = length(quench.operators)
    exps = Array{Complex128,2}(nops,steps)
    for counter = 1:steps
        time = counter*total_time/steps
        mps = (quench.uMPO(dt,time)) * mps
        mps = reduce(mps,D)
        ## expectation values:
        for k = 1:nops
            exps[k,counter] = mpoExpectation(mps.mps,quench.operators[k](time))
        end
    end
    return exps
end


function time_evolve(mps, block, total_time, steps, D, entropy_cut, params, eth, mpo=nothing)
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
    if !mpo_to_mps_trafo && check_LRcanonical(mps[1],-1) # use rightcanonical mps as default input for sweeping direction
        makeCanonical(mps)
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
                makeCanonical(mps,i+2)
            else # more efficiently for pure mps
                mps[i+1],R,DB = LRcanonical(mps[i+1],-1) # leftcanonicalize current sites
                if i < L-1 || odd_length
                    @tensor mps[i+2][-1,-2,-3] := R[-1,1]*mps[i+2][1,-2,-3]
                end
            end
        end

        ## ************ left sweep over even sites
        if !mpo_to_mps_trafo
            mps[L],R,DB = LRcanonical(mps[L],1) # rightcanonicalize at right end
            @tensor mps[L-1][:] := mps[L-1][-1,-2,1]*R[1,-3]
        end
        for i = even_start:-2:2
            W = expm(-1im*stepsize*block(i,time,params))
            W = reshape(W, (d,d,d,d))
            mps[i], mps[i+1] = block_decimation(W, mps[i], mps[i+1], D, 1)
			# preserve canonical structure:
            if mpo_to_mps_trafo
                makeCanonical(mps,i-2)
            else
                mps[i],R,DB = LRcanonical(mps[i],1) # rightcanonicalize current sites
                @tensor mps[i-1][:] := mps[i-1][-1,-2,1]*R[1,-3]
            end
        end
        if !mpo_to_mps_trafo
            mps[1],R,DB = LRcanonical(mps[1],1) # rightcanonicalize at left end
        end

        ## expectation values:
        if mpo != nothing
            if mpo == "Ising"
                J0, h0, g0 = params
                J, h, g = evolveIsingParams(J0, h0, g0, time)
                hamiltonian = IsingMPO(L, J, h, g)
                expect[counter,:] = [time mpoExpectation(mps,hamiltonian)]
            elseif mpo == "Heisenberg"
                Jx0, Jy0, Jz0, hx0 = params
                Jx, Jy, Jz, hx = evolveHeisenbergParams(Jx0, Jy0, Jz0, hx0, time)
                hamiltonian = HeisenbergMPO(L, Jx, Jy, Jz, hx)
                expect[counter,:] = [time mpoExpectation(mps,hamiltonian)]
            else
                expect[counter,:] = [time mpoExpectation(mps,mpo)]
            end
        end

        ## entanglement entropy:
        if entropy_cut > 0
            entropy[counter,:] = [time entropy(mps,entropy_cut)]
        end

		## ETH calculations:
		if eth[1] == true
			E1, hamiltonian = real(eth[2]), eth[3]
			rho = multiplyMPOs(mps,mps)
			E_thermal = real(traceMPO(multiplyMPOs(rho,hamiltonian)))
			if E_thermal <= E1
				return E_thermal, real(time*1im) # im*time = beta/2
			end
		end
    end

    return expect, entropy
end
