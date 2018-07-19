module iMPS
using TensorOperations
using MPS
using LinearMaps
using TEBD

## Has issues with truncation. Perhaps GammaLambda canonical form should be used.

function random_iMPO(d,D)
    ran = rand(d*d,D*D)+1im*rand(d*d,D*D)
    ran = ran/norm(ran)
    A = B = reshape(ran,D,d,d,D)
    return A,B
end
function random_iMPS(d,D)
    ran = rand(d,D*D)+1im*rand(d,D*D)
    ran = ran/norm(ran)
    A = B = reshape(ran,D,d,D)
    return A,B
end

function getT(A,B)
    function T(R)
        R = reshape(R, last(size(B)), last(size(B)))
        if length(size(A))==4
            @tensor ret[:] := B[-2,5,6,2]*conj(B[-1,5,6,1])*R[1,2]
            @tensor ret[:] := A[-2,5,6,2]*conj(A[-1,5,6,1])*ret[1,2]
        else
            @tensor ret[:] := B[-2,5,2]*conj(B[-1,5,1])*R[1,2]
            @tensor ret[:] := conj(A[-2,5,2]*conj(A[-1,5,1])*ret[1,2])
        end
        return reshape(ret,prod(size(ret)))
    end
    function Tc(L)
        L = conj(reshape(L, first(size(A)), first(size(A))))
        if length(size(A))==4
            @tensor ret[:] := A[2,5,6,-2]*conj(A[1,5,6,-1])*L[1,2]
            @tensor ret[:] := conj(B[2,5,6,-2]*conj(B[1,5,6,-1])*ret[1,2])
        else
            @tensor ret[:] := A[2,5,-2]*conj(A[1,5,-1])*L[1,2]
            @tensor ret[:] := conj(B[2,5,-2]*conj(B[1,5,-1])*ret[1,2])
        end
        return reshape(ret,prod(size(ret)))
    end
    T = LinearMap{Complex128}(T,Tc, first(size(A))^2, last(size(B))^2)
    return T
end

function getTL(A,B)
    function TL(L)
        L = reshape(L, first(size(A)), first(size(A)))
        if length(size(A))==4
            @tensor ret[:] := A[2,5,6,-2]*conj(A[1,5,6,-1])*L[1,2]
            @tensor ret[:] := B[2,5,6,-2]*conj(B[1,5,6,-1])*ret[1,2]
        else
            @tensor ret[:] := A[2,5,-2]*conj(A[1,5,-1])*L[1,2]
            @tensor ret[:] := B[2,5,-2]*conj(B[1,5,-1])*ret[1,2]
        end
        return reshape(ret,prod(size(ret)))
    end
    return conj(LinearMap{Complex128}(TL, last(size(B))^2, first(size(A))^2))
end

function getTR(A,B)
    function TR(R)
        R = reshape(R, last(size(B)), last(size(B)))
        if length(size(A))==4
            @tensor ret[:] := B[-2,5,6,2]*conj(B[-1,5,6,1])*R[1,2]
            @tensor ret[:] := A[-2,5,6,2]*conj(A[-1,5,6,1])*ret[1,2]
        else
            @tensor ret[:] := B[-2,5,2]*conj(B[-1,5,1])*R[1,2]
            @tensor ret[:] := A[-2,5,2]*conj(A[-1,5,1])*ret[1,2]
        end
        return reshape(ret,prod(size(ret)))
    end
    return LinearMap{Complex128}(TR, first(size(A))^2, last(size(B))^2)
end


function getTOR(A,B,opA,opB)
    function TOR(R)
        R = reshape(R, last(size(B)), last(size(B)))
        F = [1]
        if length(size(A))==4
            @tensor ret[:] := B[-2,5,6,2]*opB[-3,7,5,8]*conj(B[-1,7,6,1])*R[1,2]*F[8]
            @tensor ret[:] := A[-2,5,6,2]*opA[9,7,5,8]*conj(A[-1,7,6,1])*ret[1,2,8]*F[9]
        else
            @tensor ret[:] := B[-2,5,2]*opB[-3,6,5,8]*conj(B[-1,6,1])*R[1,2]*F[8]
            @tensor ret[:] := A[-2,5,2]*opA[9,6,5,8]*conj(A[-1,6,1])*ret[1,2,8]*F[9]
        end
        return reshape(ret,prod(size(ret)))
    end
    return LinearMap{Complex128}(TOR, first(size(A))^2, last(size(B))^2)
end
function getTOL(A,B,opA,opB)
    function TOL(L)
        L = reshape(L, first(size(A)), first(size(A)))
        F = [1]
        if length(size(A))==4
            @tensor ret[:] := A[2,5,6,-2]*opA[8,7,5,-3]*conj(A[1,7,6,-1])*L[1,2]*F[8]
            @tensor ret[:] := B[2,5,6,-2]*opB[8,7,5,9]*conj(B[1,7,6,-1])*ret[1,2,8]*F[9]
        else
            @tensor ret[:] := A[2,5,-2]*opA[8,6,5,-3]*conj(A[1,6,-1])*L[1,2]*F[8]
            @tensor ret[:] := B[2,5,-2]*opB[8,6,5,9]*conj(B[1,6,-1])*ret[1,2,8]*F[9]
        end
        return reshape(ret,prod(size(ret)))
    end
    return LinearMap{Complex128}(TOL, last(size(B))^2, first(size(A))^2)
end

function Teigs(A,B, n=1, vecs=:R,tol=1e-8)
    #T = @tensor A[-1,3,4,1]*B[1,5,6,-3]*conj(A[-2,3,4,2])*conj(B[2,5,6,-4])*R[]
    T = getT(A,B)
    if vecs == :LR
        evalsL, evecsL = lmeigs(T' ,nev=n,tol=tol)
        evalsR, evecsR = lmeigs(T ,nev=n,tol=tol)
        return evalsL,evalsR,evecsL,evecsR
    end
    if vecs == :L
        evalsL, evecsL = lmeigs(T' ,nev=n,tol=tol)
        return evalsL, evecsL
    end
    if vecs == :R
        evalsR, evecsR = lmeigs(T ,nev=n,tol=tol)
        return evalsR, evecsR
    end
    return
end

function lmeigs(T,tol=1e-8; args...)
    if prod(size(T))<10
        return eig(Base.full(T))
    else
        return eigs(T;args...,tol=tol)
    end
end

function normalize_iMPS(A,B)
    evals,evecs= Teigs(A,B,1);
    c = evals[1]^(-1/4)
    return A*c, B*c
end

function expectation(A,B,opA,opB)
    TOR = getTOR(A,B,opA,opB)
    eL,eR,vL,vR = Teigs(A,B,1,:LR)
    return Base.full(eL[1]^(-1)*(vL'*(TOR*vR))/(vL'*vR)[1])
end

function half_step(A,B,W,D;tol=0)
    mpo = length(size(A))==4
    X,Y = getChol(A,B)
    if mpo
        @tensor A[:] := X[-1,2]*A[2,-2,-3,-4]
        @tensor B[:] := Y[-4,1]*B[-1,-2,-3,1]
    else
        @tensor A[:] := X[-1,2]*A[2,-2,-3]
        @tensor B[:] := Y[-3,1]*B[-1,-2,1]
    end
    A, B, err = TEBD.block_decimation(W, A, B, D,tol=tol)
    if mpo
        @tensor A[:] := inv(X)[-1,2]*A[2,-2,-3,-4]
        @tensor B[:] := inv(Y)[-4,1]*B[-1,-2,-3,1]
    else
        @tensor A[:] := inv(X)[-1,2]*A[2,-2,-3]
        @tensor B[:] := inv(Y)[-3,1]*B[-1,-2,1]
    end
    return A,B,err
end

function itebd(A, B, hamblock, total_time, steps, D, operators=[];tol=0,increment=1)
    dt = total_time/steps
    nop = length(operators)
    opvalues = Array{Any,2}(steps,1+nop)
    total_err = zeros(steps)
    sA = size(A)
    d = sA[2]
    datacounter=0
    for i = 1:steps
        print(i," ", size(A)[1], " ",)
        time = dt*(i-1)
        W = expm(-1im*dt*hamblock(time))
        W = reshape(W, (d,d,d,d))
        A,B,err=half_step(A,B,W,D,tol=tol)
        total_err[i]+=err
        B,A,err=half_step(B,A,W,D,tol=tol)
        total_err[i]+=err

        a = maximum(abs.(A))
        b = maximum(abs.(B))
        eval = Teigs(A,B,1)[1][1]
        A = A*sqrt(b/(eval*a))
        B = B*sqrt(a/(eval*b))

        if steps%increment==0
            datacounter+=1
            opvalues[datacounter, 1] = time
            for k = 1:nop
                opvalues[datacounter, k+1] = mean([expectation(A,B,operators[k](time)[1],operators[k](time)[2]) expectation(B,A,operators[k](time)[1],operators[k](time)[2])])
            end
        end
    end
    return A,B, opvalues, total_err
end

function getChol(A,B)
    sA = size(A)
    sB = size(B)
    eL,eR,vL,vR = Teigs(A,B,1,:LR)
    vL = reshape(vL,sA[1],sA[1])
    vR = reshape(vR,last(sB),last(sB))

    lH = (vL+vL')/2
    lH = lH/trace(lH)
    rH = (vR+vR')/2
    rH = rH/trace(rH)
    X = Array(chol(Hermitian(lH)))
    Y = Array(chol(Hermitian(rH)))
    # X = sqrt(tvL)*Base.full(chol(Hermitian(vL)))
    # Y = sqrt(tvR)*Base.full(chol(Hermitian(vR)))
    # if length(size(A))==4
    #     @tensor A[:] := inv(Y)[1,-1]*X[1,2]*A[2,-2,-3,-4]
    #     @tensor B[:] := B[-1,-2,-3,2]*Y[4,2]*inv(X)[-4,4]
    # else
    #     @tensor A[:] := inv(Y)[1,-1]*X[1,2]*A[2,-2,-3]
    #     @tensor B[:] := inv(X)[3,-3]*Y[3,1]*B[-1,-2,1]
    # end
    return X,Y
end

function iIsing(J,h,g)
    opA = Array{Complex128}(1,2,2,3)
    opA[1,:,:,:] = reshape([si J*sz h*sx/2+g*sz/2],2,2,3)
    opB = Array{Complex128}(3,2,2,1)
    opB[:,:,:,1] = permutedims(reshape([h*sx/2+g*sz/2 sz si], 2,2,3), [3,1,2])
    return opA,opB
end
function iIsingBlock(J,h,g)
    return J*ZZ + h/2*(XI+IX) + g/2*(ZI+IZ)
end

end
