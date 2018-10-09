using MPS
using TensorOperations

function prepareGL(mps,Dmax,tol=0)
    N = length(mps)
    MPS.makeCanonical(mps,0)
    l = Array{Array{Complex128,2}}(N+1)
    g = Array{Array{Complex128,3}}(N)
    l[1] = eye(Complex128,1,size(mps[1])[1])
    for k = 1:N-1
        st = size(mps[k])
        tensor = reshape(mps[k],st[1]*st[2],st[3])
        U,S,V = svd(tensor)
        V=V'
        U,S,V,D,err = truncate_svd(U, S, V, Dmax,tol)
        U = reshape(U,st[1],st[2],D)
        @tensor g[k][:] := inv(l[k])[-1,1]*U[1,-2,-3]
        l[k+1] = diagm(S)
        @tensor mps[k+1][:] := l[k+1][-1,1]*V[1,2]*mps[k+1][2,-2,-3]
    end
    g[N] = mps[N]
    l[1] = eye(Complex128,size(g[1])[1])
    l[N+1] = eye(Complex128,size(g[N])[3])
    return g,l
end

function updateBlock(lL,lM,lR,gL,gR,block,Dmax,tol)
    @tensor theta[:] := lL[-1,1]*gL[1,2,4]*lM[4,5]*gR[5,3,6]*lR[6,-4]*block[-2,-3,2,3]
    D1l,d,d,D2r = size(theta)
    theta = reshape(theta, D1l*d,d*D2r)
    U,S,V = svd(theta, thin=true)
    V = V'
    D1 = size(S)[1] # number of singular values
    U,S,V,D1,err = truncate_svd(U,S,V,Dmax,tol)
    U=reshape(U,D1l,d,D1)
    V=reshape(V,D1,d,D2r)
    @tensor TL[:] := inv(lL)[-1,1]*U[1,-2,-3]
    @tensor TR[:] := V[-1,-2,3]*inv(lR)[3,-3]

    return TL,diagm(S),TR, err
end

function localOpExp(g,l,op,site)
    @tensor theta[:] := l[site][-1,1]*g[site][1,-2,3]*l[site+1][3,-3]
    @tensor r[:] :=theta[1,2,3]*op[4,2]*conj(theta[1,4,3])
    return r[1]
end
function gl_mpoExp(g,l,mpo)
    N = length(g)
    F = Array{Complex128}(1,1,1)
    F[1,1,1]=1
    @tensor F[-1,-2,-3] :=ones(1)[-2]*l[1][a,-3]*conj(l[1][a,-1])
    for k = 1:N
        @tensor F[:] := F[1,2,3]*g[k][3,5,-3]*mpo[k][2,4,5,-2]*conj(g[k][1,4,-1])
        @tensor F[:] := F[1,-2,3]*l[k+1][3,-3]*conj(l[k+1][1,-1])
    end
    @tensor F[:] := F[1,-2,3]*l[N+1][3,2]*conj(l[N+1][1,2])
    return F[1]
end

function gl_quench(N,time,steps,maxD,tol,inc)
    hamblocks(time) = TEBD.isingHamBlocks(N,1,0,0)
    opEmpo = MPS.IsingMPO(N,1,0,0)
    opE(time,g,l) = gl_mpoExp(g,l,opEmpo)
    opmag(time,g,l) = localOpExp(g,l,sx,Int(floor(N/2)))
    opnorm(time,g,l) =  gl_mpoExp(g,l,MPS.IdentityMPO(N,2))
    ops = [opE opmag opnorm]
    mps = MPS.randomMPS(N,2,maxD)
    g,l = prepareGL(mps,maxD,tol)
    println(gl_mpoExp(g,l,MPS.IdentityMPO(N,2)))
    return gl_tebd(g,l,hamblocks,time,steps,maxD,ops,tol=tol,increment=inc)
end

function gl_tebd(g,l, hamblocks, total_time, steps, D, operators; tol=0, increment=1)
    ### block = hamiltonian
    ### use -im*total_time for imaginary time evolution
    ### assumption: start with rightcanonical mps
	### eth = (true,E1,hamiltonian) --> do ETH calcs if true for excited energy E1 wrt hamiltonian
    stepsize = total_time/steps
    nop = length(operators)
    opvalues = Array{Any,2}(Int(floor(steps/increment)),nop+1)
    err = Array{Any,1}(steps)
    datacount=1
    for counter = 1:steps

        time = counter*total_time/steps
        err[counter] = gl_tebd_step(g,l,hamblocks(time),stepsize,D,tol=tol)
        if counter % increment == 0
            println("step ",counter," / ",steps)
            opvalues[datacount,1] = time
            for k = 1:nop
                opvalues[datacount,k+1] = operators[k](time,g,l)
            end
            datacount+=1
        end
    end

    return opvalues, err
end

function gl_tebd_step(g,l, hamblocks, dt, D; tol=0)
    d = size(g[1])[2]
    N = length(mps)
    total_error = 0
    for k = 1:2:N-1
        W = expm(-1im*dt*hamblocks[k])
        W = reshape(W, (d,d,d,d))
        g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W, D, tol)
        total_error += error
    end
    for k = 2:2:N-3
        W = expm(-1im*dt*hamblocks[k])
        W = reshape(W, (d,d,d,d))
        g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W, D, tol)
        total_error += error
    end
    return total_error
end

function truncate_svd(U, S, V, D,tol=0)
    Dtol = 0
    tot = sum(S.^2)
    while (Dtol+1 <= length(S)) && sum(S[Dtol+1:end].^2)/tot>=tol
        Dtol+=1
    end
    D = min(D,Dtol)
    err = sum(S[D+1:end].^2)
    U = U[:, 1:D]
    S = S[1:D]
    S = S/sqrt(sum(S.^2))
    V = V[1:D, :]
    return U, S, V, D,err
end
