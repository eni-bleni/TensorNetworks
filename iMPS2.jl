module imps2
using TensorOperations
using LinearMaps
struct iMPS2
    A
    B
    S
    SM
end
## Some unfinished work for optimizing imps

function random_A_B(d,D)
    ran = rand(d,D*D)+1im*rand(d,D*D)
    ran = ran/norm(ran)
    A = B = reshape(ran,D,d,D)
    return A,B
end

function imps_from_A_B(A,B)
    T = getT(A,B)
    eR,vR = eigs(T,nev=1)
    eL,vL = eigs(T',nev=1) #eig of T'
    vRmat = reshape(vR,Int(sqrt(length(vR))),Int(sqrt(length(vR))))
    vLmat = reshape(vL,Int(sqrt(length(vL))),Int(sqrt(length(vL))))
    l = Hermitian(vLmat+vLmat')
    r = Hermitian(vRmat+vRmat')
    trl = abs(trace(l*r))
    l = l/sqrt(trl)
    r = r/sqrt(trl)
    xl = sqrtm(l)
    xr = sqrtm(r)
    ixl = inv(xl)
    ixr = inv(xr)
    @tensor S[:] := xl[-1,1]*xr[1,-2]
    @tensor A[:] := ixr[-1,1]*A[1,-2,-3]
    @tensor B[:] := B[-1,-2,3]*ixl[3,-3]
    U,S,Vt = svd(S)

    @tensor A[:] := Vt'[-1,1]*A[1,-2,-3]
    @tensor B[:] := A[-1,-2,3]*U[3,-3]
    @tensor blob[:] := A[-1,-2,3]*B[3,-3,-4]
    sa = size(A)
    sb = size(B)
    blob = reshape(blob,sa[1]*sa[2],sb[2]*sb[3])
    U,SM,Vt = svd(blob)
    d = length(SM)
    A = reshape(U,sa[1],sa[2],d)
    B = reshape(Vt,d,sb[2],sb[3])

    return iMPS2(A,B,S,SM)
end

function canonicalize(imps)

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
    return LinearMap{Complex128}(T,Tc, first(size(A))^2, last(size(B))^2)
end

function getTOR(A,B,opA,opB)
    function TOR(R)
        R = reshape(R, last(size(B)), last(size(B)))
        F = [1]
        @tensor ret[:] := B[-2,5,2]*opB[-3,6,5,8]*conj(B[-1,6,1])*R[1,2]*F[8]
        @tensor ret[:] := A[-2,5,2]*opA[9,6,5,8]*conj(A[-1,6,1])*ret[1,2,8]*F[9]
        return reshape(ret,prod(size(ret)))
    end
    return LinearMap{Complex128}(TOR, first(size(A))^2, last(size(B))^2)
end

end
