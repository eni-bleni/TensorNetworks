using MPS
using TensorOperations

function prepareGL(mps,Dmax,tol=0)
    N = length(mps)
    MPS.makeCanonical(mps,0)

    trafo = false
    if length(size(mps[1]))==4
        d = size(mps[1])[2]
        mps = mpo_to_mps(mps)
        trafo=true
    end
    l = Array{Array{Complex128,2}}(N+1)
    g = Array{Array{Complex128,3}}(N)
    l[1] = spdiagm(ones(Complex128,size(mps[1])[1]))
    for k = 1:N-1
        st = size(mps[k])
        tensor = reshape(mps[k],st[1]*st[2],st[3])
        U,S,V = svd(tensor)
        V=V'
        U,S,V,D,err = truncate_svd(U, S, V, Dmax,tol)
        U = reshape(U,st[1],st[2],D)
        @tensor g[k][:] := inv(l[k])[-1,1]*U[1,-2,-3]
        l[k+1] = spdiagm(S)
        @tensor mps[k+1][:] := l[k+1][-1,1]*V[1,2]*mps[k+1][2,-2,-3]
    end
    st = size(mps[N])
    Q,R = qr(reshape(mps[N],st[1]*st[2],st[3]))
    @tensor g[N][:] := inv(l[N])[-1,1]*reshape(Q,st[1],st[2],size(Q)[2])[1,-2,-3]
    l[N+1] = sparse(R)
    l[1] = spdiagm(ones(Complex128,size(g[1])[1]))
    # l[N+1] = eye(Complex128,size(g[N])[3])
    if trafo
        g2 = Array{Array{Complex128,4}}(N)
        for k=1:N
            g2[k] = reshape(g[k],size(g[k])[1],d,d,size(g[k])[3])
        end
        g=g2
    end

    return g,l
end

function gl_to_mps(g,l)
    N = length(g)
    mps = Array{Array{Complex128,3}}(N)
    for k = 1:N
        @tensor mps[k][:] := l[k][-1,1]*g[k][1,-2,-3]
    end
    @tensor mps[N][:] := mps[N][-1,-2,3]*l[N+1][3,-3]
    return mps
end

function mpo_to_mps(mpo)
    N = length(mpo)
    mps = Array{Array{Complex128,3}}(N)
    for k = 1:N
        s = size(mpo[k])
        mps[k] = reshape(mpo[k],s[1],s[2]*s[3],s[4])
    end
    return mps
end
function mps_to_mpo(mps)
    N = length(mps)
    s = size(mps)
    d = Int(sqrt(s[2]))
    mpo = Array{Array{Complex128,4}}(N)
    for k = 1:N
        mpo[k] = reshape(mps,s[1],d,d,s[3])
    end
    return mpo
end

function sparse_l(g,l,dir=:left)
    sg=size(g)
    l=sparse(l)
    if dir == :left
        if length(sg)==3
            g = reshape(g,sg[1],sg[2]*sg[3])
            A = l*g
        else
            g = reshape(g,sg[1],sg[2]*sg[3]*sg[4])
            A = l*g
        end
    elseif dir==:right
        if length(sg)==3
            g = reshape(g,sg[1]*sg[2],sg[3])
            A = g*l
        else
            g = reshape(g,sg[1]*sg[2]*sg[3],sg[4])
            A = g*l
        end
    end
    return reshape(A,sg)
end

function updateBlock(lL,lM,lR,gL,gR,block,Dmax,tol)
    gL = sparse_l(gL,lL,:left)
    gL = sparse_l(gL,lM,:right)
    gR = sparse_l(gR,lR,:right)
    if length(size(gL))==4
        @tensor theta[:] := gL[-1,2,-2,5]*gR[5,3,-5,-6]*block[-4,-3,3,2]
        #slow @tensor theta[:] := lL[-1,1]*gL[1,-2,-4,4]*lM[4,-3]
        #slow @tensor theta[:] := theta[-1,2,5,-2]*gR[5,3,-5,6]*lR[6,-6]*block[-4,-3,3,2]
        st = size(theta)
        theta = reshape(theta,st[1]*st[2],st[3],st[4],st[5]*st[6])
    else
        @tensor theta[:] := gL[-1,2,5]*gR[5,3,-4]*block[-3,-2,3,2]
        #slow  @tensor theta[:] := lL[-1,1]*gL[1,-2,4]*lM[4,-3]
        #slow @tensor theta[:] := theta[-1,2,5]*gR[5,3,6]*lR[6,-4]*block[-3,-2,3,2]
        #slowest @time @tensor theta[:] := lL[-1,1]*gL[1,2,4]*lM[4,5]*gR[5,3,6]*lR[6,-4]*block[-2,-3,2,3]
    end

    D1l,d,d,D2r = size(theta)
    theta = reshape(theta, D1l*d,d*D2r)
    U,S,V = svd(theta, thin=true)
    V = V'
    D1 = size(S)[1] # number of singular values
    U,S,V,D1,err = truncate_svd(U,S,V,Dmax,tol)
    if length(size(gL))==4
        U=reshape(U,st[1],st[2],d,D1)
        V=reshape(V,D1,d,st[5],st[6])
        ilL = spdiagm(1./diag(lL))
        ilR = spdiagm(1./diag(lR))
        U = permutedims(sparse_l(U,ilL,:left),[1 3 2 4])
        V = sparse_l(V,ilR,:right)
        # @tensor U[:] := inv(lL)[-1,1]*U[1,-3,-2,-4]
        # @tensor V[:] := V[-1,-2,-3,3]*inv(lR)[3,-4]
    else
        U=reshape(U,D1l,d,D1)
        V=reshape(V,D1,d,D2r)
        ilL = spdiagm(1./diag(lL))
        ilR = spdiagm(1./diag(lR))
        U = sparse_l(U,ilL,:left)
        V = sparse_l(V,ilR,:right)
        # @tensor U[:] := inv(lL)[-1,1]*U[1,-2,-3]
        # @tensor V[:] := V[-1,-2,3]*inv(lR)[3,-3]
    end

    return U,spdiagm(S),V, err
end

function localOpExp(g,l,op,site)
    theta = sparse_l(g[site],l[site],:left)
    theta = sparse_l(theta,l[site+1],:right)
    if length(size(g[1]))==3
        # @tensor theta[:] := l[site][-1,1]*g[site][1,-2,3]*l[site+1][3,-3]
        @tensor r[:] :=theta[1,2,3]*op[4,2]*conj(theta[1,4,3])
    elseif length(size(g[1]))==4
        # @tensor theta[:] := l[site][-1,1]*g[site][1,-2,-4,3]*l[site+1][3,-3]
        @tensor r[:] :=theta[1,2,3,5]*op[4,2]*conj(theta[1,4,3,5])
    end
    return r[1]
end
function gl_mpoExp(g,l,mpo)
    N = length(g)
    F = Array{Complex128}(1,1,1)
    F[1,1,1]=1
    @tensor F[-1,-2,-3] :=ones(1)[-2]*l[1][a,-3]*conj(l[1][a,-1])
    for k = 1:N
        if length(size(g[1]))==3
            @tensor F[:] := F[1,2,3]*g[k][3,5,-3]*mpo[k][2,4,5,-2]*conj(g[k][1,4,-1])
        elseif length(size(g[1]))==4
            @tensor F[:] := F[1,2,3]*g[k][3,5,6,-3]*mpo[k][2,4,5,-2]*conj(g[k][1,4,6,-1])
        end
        @tensor F[:] := F[1,-2,3]*l[k+1][3,-3]*conj(l[k+1][1,-1])
    end
    @tensor F[:] := F[1,-1,1]
    return F[1]
end

function isingHamBlocks(L,J,h,g)
    blocks = Array{Any,1}(L)
    for i=1:L
        if i==1
            blocks[i] = J*ZZ + h/2*(2XI+IX) + g/2*(2*ZI+IZ)
        elseif i==L-1
            blocks[i] = J*ZZ + h/2*(XI+2IX) + g/2*(ZI+2*IZ)
        else
            blocks[i] = J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ)
        end
    end
    return blocks
end

function ops_on_gl(g,l,ops)
    N = length(g)
    fourlegs=false
    if length(size(g[1]))==4
        fourlegs=true
    end
    for k=1:N
        if fourlegs
            @tensor g[k][:] := g[k][-1,2,-3,-4]*ops[k][-2,2]
        else
            @tensor g[k][:] := g[k][-1,2,-3]*ops[k][-2,2]
        end
    end
    return g
end

function check_canon(g,l)
    N = length(g)
    for k=1:N
        @tensor D[:]:= l[k][1,2]*g[k][2,3,-2]*conj(l[k][1,4])*conj(g[k][4,3,-1])
        println("L: ",real(det(D)),"_",real(trace(D))/size(D)[1])
        @tensor D[:]:= l[k+1][2,1]*g[k][-2,3,2]*conj(l[k+1][4,1])*conj(g[k][-1,3,4])
        println("R: ",real(det(D)),"_",real(trace(D))/size(D)[1])
    end
end
function gl_quench(N,time,steps,maxD,tol,inc)
    hamblocksTH(time) = isingHamBlocks(N,1,1,0)
    hamblocks(time) = isingHamBlocks(N,1,1,0)
    opEmpo = MPS.IsingMPO(N,1,1,0)
    opE(time,g,l) = gl_mpoExp(g,l,opEmpo)
    opmag(time,g,l) = localOpExp(g,l,sx,Int(floor(N/2)))
    opnorm(time,g,l) = gl_mpoExp(g,l,MPS.IdentityMPO(N,2))
    ops = [opE opmag opnorm]
    mpo = MPS.IdentityMPO(N,2)
    # mps = mpo_to_mps(mpo)
    mps = MPS.randomMPS(N,2,5)
    g,l = prepareGL(mpo,maxD)
    # check_canon(g,l)
    opvals, err = gl_tebd(g,l,hamblocksTH,-20*im,1000,maxD,ops,tol=tol,increment=inc)
    # check_canon(g,l)
    # println(l[5])
    pert_ops = fill(expm(1e-3*im*sx),N)
    ops_on_gl(g,l,pert_ops)
    opvals, err = gl_tebd(g,l,hamblocks,time,steps,maxD,ops,tol=tol,increment=inc)
    # check_canon(g,l)
    return opvals, err, size(g[Int(floor(N/2))])
end

function gl_tebd(g,l, hamblocks, total_time, steps, D, operators; tol=0, increment=1, thermal=false)
    ### block = hamiltonian
    ### use -im*total_time for imaginary time evolution
    ### assumption: start with rightcanonical mps
	### eth = (true,E1,hamiltonian) --> do ETH calcs if true for excited energy E1 wrt hamiltonian
    stepsize = total_time/steps
    nop = length(operators)
    opvalues = SharedArray{Complex128,2}(Int(floor(steps/increment)),nop+1)
    err = SharedArray{Complex128,1}(steps)
    datacount=0
    for counter = 1:steps
        time = counter*total_time/steps
        # if thermal
        #     time = t_th[counter]
        err[counter] = gl_tebd_step(g,l,hamblocks(time),stepsize,D,tol=tol)
        if counter % increment == 0
            datacount+=1
            println("step ",counter," / ",steps)
            opvalues[datacount,1] = time
            @sync @parallel for k = 1:nop
                opvalues[datacount,k+1] = operators[k](time,g,l)
            end
        end
    end
    return opvalues, err
end

function gl_tebd_step(g,l, hamblocks, dt, D; tol=0)
    d = size(g[1])[2]
    N = length(g)
    total_error = 0
    W = reshape.(expm.(-1im*dt*hamblocks),d,d,d,d)
    if imag(dt)==0
        @sync @parallel for k = 1:2:N-1
        # W = expm(-1im*dt*hamblocks[k])
        # W = reshape(Ws[k], (d,d,d,d))
        g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol)
        total_error += error
        end
        s = isodd(N) ? N-1 : N-2
        @sync @parallel for k = s:-2:1
            # W = expm(-1im*dt*hamblocks[k])
            # W = reshape(Ws[k], (d,d,d,d))
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol)
            total_error += error
        end
    else
        for k = 1:N-1
            # W = expm(-1im*dt*hamblocks[k])
            # W = reshape(Ws[k], (d,d,d,d))
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol)
            total_error += error
        end
        for k = N-1:-1:1
            WI = reshape(II, (d,d,d,d))
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],WI, D, tol)
            total_error += error
        end
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
