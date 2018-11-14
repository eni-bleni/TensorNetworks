using TensorOperations
using SparseArrays
using LinearAlgebra
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
    l[1] = diagm(ones(Complex128,size(mps[1])[1]))
    for k = 1:N-1
        st = size(mps[k])
        tensor = reshape(mps[k],st[1]*st[2],st[3])
        U,S,V = svd(tensor)
        V=Array(V')
        U,S,V,D,err = truncate_svd(U, S, V, Dmax,tol)
        U = reshape(U,st[1],st[2],D)
        ilk=inv(l[k])
        println(typeof(ilk),typeof(U))
        @tensor g[k][:] := ilk[-1,1]*U[1,-2,-3]
        l[k+1] = diagm(S)
        @tensor mps[k+1][:] := l[k+1][-1,1]*V[1,2]*mps[k+1][2,-2,-3]
    end
    st = size(mps[N])
    Q,R = qr(reshape(mps[N],st[1]*st[2],st[3]))
    Q=Array(Q)
    R=Array(R)
    ilN=inv(l[N])
    @tensor g[N][:] := ilN[-1,1]*reshape(Q,st[1],st[2],size(Q)[2])[1,-2,-3]
    l[N+1] = diagm(ones(Complex128,size(g[1])[1]))
    l[1] = diagm(ones(Complex128,size(g[1])[1]))
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
function gl_to_mpo(g,l)
    N = length(g)
    mpo = Array{Array{Complex128,4}}(N)
    for k = 1:N
        @tensor mpo[k][:] := l[k][-1,1]*g[k][1,-2,-3,-4]
    end
    @tensor mpo[N][:] := mpo[N][-1,-2,-3,3]*l[N+1][3,-4]
    return mpo
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
    l2=sparse(l)
    if dir == :left
        if length(sg)==3
            g3 = reshape(g,sg[1],sg[2]*sg[3])
            A = l2*g3
        else
            g4 = reshape(g,sg[1],sg[2]*sg[3]*sg[4])
            A = l2*g4
        end
    elseif dir==:right
        if length(sg)==3
            g3 = reshape(g,sg[1]*sg[2],sg[3])
            A = g3*l2
        else
            g4 = reshape(g,sg[1]*sg[2]*sg[3],sg[4])
            A = g4*l2
        end
    end
    return reshape(A,sg)
end

function updateBlocktsvd(lL,lM,lR,gL,gR,block,Dmax,tol)
    gL = sparse_l(gL,lL,:left)

    if length(size(gL))==4
        sgl = size(gL)
        initU = reshape(permutedims(gL,[1 3 2 4]),sgl[1]*sgl[2]*sgl[3],sgl[4])
    else
        sgl = size(gL)
        initU = reshape(permutedims(gL),sgl[1]*sgl[2],sgl[3])
    end
    gL = sparse_l(gL,lM,:right)
    gR = sparse_l(gR,lR,:right)
    sr = size(gR)
    sl = size(gL)
    sb = size(block)
    if length(sl)==4
        function theta(vec)
            vec = reshape(vec,sb[1],sr[3],sr[4])
            @tensor out[:] := gR[-1,-2,7,6]*vec[-3,7,6]
            @tensor out[:] := gL[-1,4,-3,1]*block[3,-2,2,4]*out[1,2,3]
            # @tensoropt (5,6) out[:] := gL[-1,2,-3,5]*gR[5,3,7,6]*block[4,-2,3,2]*vec[4,7,6]
            return reshape(out,sl[1]*sb[2]*sl[3])
        end
        function thetaconj(vec)
            vec = reshape(vec,sl[1],sb[2],sl[3])
            @tensoropt (1,5) out[:] := gL[1,2,4,5]*gR[5,3,-2,-3]*block[-1,8,3,2]*conj(vec[1,8,4])
            return reshape(conj(out),sb[1]*sr[3]*sr[4])
        end
        thetalin = LinearMap{Complex128}(theta,thetaconj,sb[2]*sl[3]*sl[1], sb[1]*sr[3]*sr[4])
    # elseif length(sl)==3
    #     function theta(vec)
    #         reshape(vec,sb[1],sr[3])
    #         @tensor out[:] = gL[-2,2,5]*gR[5,3,4]*block[6,-1,3,2]*vec[6,4]
    #         return reshape(out,sb[2],sl[1])
    #     end
    #     function thetaconj(vec)
    #         println("ASDSAD")
    #         reshape(vec,sb[2],sl[1])
    #         @tensor out[:] = gL[4,2,5]*gR[5,3,-2]*block[-1,6,3,2]*vec[6,4]
    #         return reshape(out,sb[1],sr[3])
    #     end
    #     thetalin = LinearMap{Complex128}(theta, sb[2]*sl[1], sb[1]*sr[3])
    # end
    end

    if min(size(thetalin)...)<2*Dmax
        U,S,V = svd(Base.full(thetalin), thin=true)
    else
        println("ASD")
        U,S,V = tsvd(thetalin, maxiter=1000,min(size(thetalin)[1],size(thetalin)[2],Dmax),tolconv=tol*100,tolreorth=tol*100)
    end
    V = V'
    D1 = size(S)[1] # number of singular values
    U,S,V,D1,err = truncate_svd(U,S,V,Dmax,tol)
    if length(size(gL))==4
        U=reshape(U,sl[1],sl[2],sl[3],D1)
        V=reshape(V,D1,sr[2],sr[3],sr[4])
        ilL = spdiagm(1./diag(lL))
        ilR = spdiagm(1./diag(lR))
        U = sparse_l(U,ilL,:left)
        V = sparse_l(V,ilR,:right)
        # @tensor U[:] := inv(lL)[-1,1]*U[1,-3,-2,-4]
        # @tensor V[:] := V[-1,-2,-3,3]*inv(lR)[3,-4]
    else
        U=reshape(U,sl[1],sl[2],D1)
        V=reshape(V,D1,sl[2],sr[3])
        ilL = spdiagm(1./diag(lL))
        ilR = spdiagm(1./diag(lR))
        U = sparse_l(U,ilL,:left)
        V = sparse_l(V,ilR,:right)
        # @tensor U[:] := inv(lL)[-1,1]*U[1,-2,-3]
        # @tensor V[:] := V[-1,-2,3]*inv(lR)[3,-3]
    end

    return U,spdiagm(S),V, err
end

function updateBlock(lL,lM,lR,gL,gR,block,Dmax,tol;counter_evo=false)
    gL = sparse_l(gL,lL,:left)
    gL = sparse_l(gL,lM,:right)
    gR = sparse_l(gR,lR,:right)
    if length(size(gL))==4
        if counter_evo
            # @tensor blob[-1,-2,-3,-4,-5,-6] := W[2,6,-2,-4]*Tl[-1,2,3,4]*conj(W[3,5,-3,-5])*Tr[4,6,5,-6]
            @tensor theta[:] := gL[-1,2,7,5]*gR[5,3,8,-6]*block[-4,-3,3,2]*conj(block[-5,-2,8,7])
        else
            # @tensor blob[-1,-2,-3,-4,-5,-6] := Tl[-1,2,-3,4]*block[2,6,-2,-4]*Tr[4,6,-5,-6]
            @tensor theta[:] := gL[-1,2,-2,5]*gR[5,3,-5,-6]*block[-4,-3,3,2]
        end

        # @tensor theta[:] := gL[-1,2,-2,5]*gR[5,3,-5,-6]*block[-4,-3,3,2]

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
    F = Array{Complex128,3}
    # F[1][1,1,1]=1
    @tensor F[-1,-2,-3] :=ones(1)[-2]*l[1][a,-3]*conj(l[1][a,-1])
    for k = 1:N
        if length(size(g[1]))==3
            @tensor F[-1,-2,-3] := F[1,2,3]*g[k][3,5,-3]*mpo[k][2,4,5,-2]*conj(g[k][1,4,-1])
        elseif length(size(g[1]))==4
            # @tensor F[:] := F[-1,2,3]*g[k][3,5,-6,-3]*mpo[k][2,-4,5,-2]
            # @tensor F[:] := F[1,-2,-3,4,6]*conj(g[k][1,4,6,-1])
            @tensor F[-1,-2,-3] := F[1,2,3]*g[k][3,5,6,-3]*mpo[k][2,4,5,-2]*conj(g[k][1,4,6,-1])
        end
        F=sparse_l(F,l[k+1],:right)
        F=sparse_l(F,l[k+1],:left)
        # @tensor F[:] := F[1,-2,3]*l[k+1][3,-3]*conj(l[k+1][1,-1])
    end
    @tensor F[:] := F[1,-1,1]
    return F[1]
end

function gl_scalarprod(gA,lA,gB,lB;flipB=false)
    N = length(gA)
    F = Array{Complex128,3}
    # F[1][1,1,1]=1
    @tensor F[-1,-2] := lA[1][a,-2]*conj(lB[1][a,-1])
    for k = 1:N
        if length(size(gA[1]))==3
            @tensor F[-1,-2] := F[1,2]*gA[k][2,5,-2]*conj(gB[k][1,5,-1])
        elseif length(size(gA[1]))==4
            # @tensor F[:] := F[-1,2,3]*g[k][3,5,-6,-3]*mpo[k][2,-4,5,-2]
            # @tensor F[:] := F[1,-2,-3,4,6]*conj(g[k][1,4,6,-1])
            if flipB
                @tensor F[-1,-2] := F[1,2]*gA[k][2,5,6,-2]*conj(gB[k][1,5,6,-1])
            else
                @tensor F[-1,-2] := F[1,2]*gA[k][2,5,6,-2]*conj(gB[k][1,5,6,-1])
            end
        end
        @tensor F[:] := F[1,2]*lA[k+1][2,-2]*conj(lB[k+1][1,-1])
    end
    @tensor F[:] := F[1,1]
    return F[1]
end

function isingHamBlocks(L,J,h,g)
    blocks = Array{Array{Complex128,2},1}(L)
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

#UNFINISHED
function mpo_on_gl(g,l,mpo)
    N = length(g)-1
    glmpo = gl_to_mpo(g,l)
    return multiplyMPOs(mpo,glmpo)
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

function gl_quench(N,time,steps,maxD,tol,inc::Int64)
    J0=1
    h0=1
    g0=0
    q = 2*pi*(3/(N-1))
    hamblocksTH(time) = isingHamBlocks(N,J0,h0,g0)
    hamblocks(time) = isingHamBlocks(N,J0,h0,g0)
    opEmpo = MPS.IsingMPO(N,J0,h0,g0)
    opE(time,g,l) = gl_mpoExp(g,l,opEmpo)
    opmag(time,g,l) = localOpExp(g,l,sx,Int(floor(N/2)))
    opnorm(time,g,l) = gl_mpoExp(g,l,MPS.IdentityMPO(N,2))
    ops = [opE opmag opnorm]
    mpo = MPS.IdentityMPO(N,2)
    # mps = mpo_to_mps(mpo)
    mps = MPS.randomMPS(N,2,5)
    g,l = prepareGL(mpo,maxD)
    # check_canon(g,l)
    # pert_ops = fill(expm(1e-3*im*sx),N)
    pert_ops = [expm(1e-3*sx*im*x) for x in sin.(q*(-1+(1:N)))]
    @time opvals, err = gl_tebd(g,l,hamblocksTH,-2*im,10,maxD,ops,tol=tol,increment=inc,st2=true)
    # check_canon(g,l)
    # println(l[5])
    ops_on_gl(g,l,pert_ops)
    @time opvals, err = gl_tebd(g,l,hamblocks,time,steps,maxD,ops,tol=tol,increment=inc,st2=true)
    # check_canon(g,l)
    return opvals, err, size(g[Int(floor(N/2))])
end

function gl_tebd(g,l, hamblocks, total_time, steps, D, operators; tol=0, increment::Int64=1, st2::Bool=false)
    ### block = hamiltonian
    ### use -im*total_time for imaginary time evolution
    ### assumption: start with rightcanonical mps
	### eth = (true,E1,hamiltonian) --> do ETH calcs if true for excited energy E1 wrt hamiltonian
    stepsize = total_time/steps
    nop = length(operators)
    opvalues = Array{Complex128,2}(1+Int(floor(steps/increment)),nop+1)
    err = Array{Complex128,1}(steps)
    datacount=1
    opvalues[datacount,1] = 0
    for k = 1:nop
        opvalues[datacount,k+1] = operators[k](time,g,l)
    end
    for counter = 1:steps
        time = counter*total_time/steps
        if !st2
            err[counter] = gl_tebd_step(g,l,hamblocks(time),stepsize,D,tol=tol)
        elseif st2
            err[counter] =  gl_tebd_step_st2(g,l,hamblocks(time),stepsize,D,tol=tol)
        end
        if counter % increment == 0
            datacount+=1
            println("step ",counter," / ",steps)
            opvalues[datacount,1] = time
            for k = 1:nop
                opvalues[datacount,k+1] = operators[k](time,g,l)
            end
        end
    end
    return opvalues, err
end

function gl_ct!(g)
    N = length(g)
    for k=1:N
        g[k]=permutedims(conj(g[k]),[1 3 2 4])
    end
end
function gl_tebd_c(gA,lA,gB,lB, hamblocks, total_time, steps, D; tol=0, increment::Int64=1, st2::Bool=false)
    ### block = hamiltonian
    ### use -im*total_time for imaginary time evolution
    ### assumption: start with rightcanonical mps
	### eth = (true,E1,hamiltonian) --> do ETH calcs if true for excited energy E1 wrt hamiltonian

    stepsize = total_time/steps
    opvalues = Array{Complex128,1}(1+Int(floor(steps/increment)))
    t = Array{Float64,1}(1+Int(floor(steps/increment)))
    errA = Array{Complex128,1}(steps)
    errB = Array{Complex128,1}(steps)

    datacount=1
    t[datacount] = 0
    opvalues[datacount] = gl_scalarprod(gA,lA,gB,lB,flipB=true)

    for counter = 1:steps
        time = counter*total_time/steps
        if !st2
            errA[counter] = gl_tebd_step(gA,lB,hamblocks(time),stepsize,D,tol=tol, counter_evo=true)
            errB[counter] = gl_tebd_step(gB,lB,hamblocks(time),-stepsize,D,tol=tol, counter_evo=true)
        elseif st2
            errA[counter] =  gl_tebd_step_st2(gA,lA,hamblocks(time),stepsize,D,tol=tol, counter_evo=true)
            errB[counter] =  gl_tebd_step_st2(gB,lB,hamblocks(time),-stepsize,D,tol=tol, counter_evo=true)
        end
        if counter % increment == 0
            datacount+=1
            println("step ",counter," / ",steps)
            t[datacount] = 2*time
            opvalues[datacount] = gl_scalarprod(gA,lA,gB,lB,flipB=true)
        end
    end
    return opvalues, errA,errB, t
end

function gl_tebd_step(g,l, hamblocks, dt, D; tol=0,counter_evo=false)
    d = size(g[1])[2]
    N = length(g)
    total_error = 0
    W = reshape.(expm.(-1im*dt*hamblocks),d,d,d,d)
    # function local_update(k)
    #     return (k,updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol)...)
    # end
    if imag(dt)==0
        # results = pmap(local_update,1:2:N-1)
        # for r in results
        #     g[r[1]]=r[2]
        #     l[r[1]+1]=r[3]
        #     g[r[1]+1]=r[4]
        #     total_error+=r[5]
        # end
        # s = isodd(N) ? N-1 : N-2
        # results = pmap(local_update,s:-2:1)
        # for r in results
        #     g[r[1]]=r[2]
        #     l[r[1]+1]=r[3]
        #     g[r[1]+1]=r[4]
        #     total_error+=r[5]
        # end
        Threads.@threads for k = 1:2:N-1
        # W = expm(-1im*dt*hamblocks[k])
        # W = reshape(Ws[k], (d,d,d,d))
        g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol,counter_evo=counter_evo)
        total_error += error
        end
        s = isodd(N) ? N-1 : N-2
        Threads.@threads for k = s:-2:1
            # W = expm(-1im*dt*hamblocks[k])
            # W = reshape(Ws[k], (d,d,d,d))
        g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol,counter_evo=counter_evo)

            total_error += error
        end
    else
        WI = reshape(II, (d,d,d,d))
        for k = 1:N-1
            # W = expm(-1im*dt*hamblocks[k])
            # W = reshape(Ws[k], (d,d,d,d))
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
        for k = N-1:-1:1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],WI, D, tol,counter_evo=counter_evo)
            total_error += error
        end
    end
    return total_error
end

 function gl_tebd_step_st2(g,l, hamblocks, dt, D; tol=0, counter_evo=false)
    d = size(g[1])[2]
    N = length(g)
    total_error = 0
    W = reshape.(expm.(-1im*dt*hamblocks),d,d,d,d)
    W2 = reshape.(expm.(-1/2*im*dt*hamblocks),d,d,d,d)
    if imag(dt)==0
        Threads.@threads for k = 1:2:N-1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W2[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
        s = isodd(N) ? N-1 : N-2
        Threads.@threads for k = s:-2:1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
        Threads.@threads for k = 1:2:N-1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W2[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
    else
        WI = reshape(II, (d,d,d,d))
        s = isodd(N) ? N-1 : N-2
        for k=1:N
            W[k] = isodd(k) ? W[k] : WI
            W2[k] = iseven(k) ? W2[k] : WI
        end
        for k = 1:N-1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W2[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
        for k = s:-1:1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
        for k = 1:N-1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],W2[k], D, tol, counter_evo=counter_evo)
            total_error += error
        end
        for k = N-1:-1:1
            g[k], l[k+1], g[k+1], error = updateBlock(l[k],l[k+1],l[k+2],g[k],g[k+1],WI, D, tol, counter_evo=counter_evo)
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
