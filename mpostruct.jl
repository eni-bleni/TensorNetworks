using TensorOperations
import Base: *, transpose, ctranspose, norm, getindex,length,
 split, promote_rule, similar

# struct MPOtensor <: AbstractArray{Complex128, 4}
#     tensor :: Array{Complex128,4}
#     getelem
# end
# Base.size(x::MPOtensor) = size(x.tensor)
# Base.IndexStyle(::Type{<:MPOtensor}) = IndexCartesian()
# Base.getindex(x::MPOtensor, elem) = getindex(x.tensor,elem)
# Base.similar(x, ::Type{MPOtensor}) = similar(x.tensor, MPOtensor, size(x))
#
# function convert(::Type{MPOtensor},tensor::Array{Complex128,4})
#     f(x...) = getindex(tensor,x...)
#     return MPOtensor(tensor,f)
# end

struct MPO
    mpo :: Array{Array{Complex128,4}}
end

struct MPS
    mps :: Array{Array{Complex128,3},1}
    dir
end
getindex(A::MPO,elems...) = getindex(A.mpo,elems...)
getindex(v::MPS,elems...) = getindex(v.mps,elems...)
Base.setindex!(A::MPO,X,elems...) = setindex!(A.mpo,X,elems...)
Base.setindex!(v::MPS,X,elems...) = setindex!(v.mps,X,elems...)
length(v::MPS) = length(v.mps)
length(A::MPO) = length(A.mpo)
Base.similar(A::MPO) = MPO(similar(A.mpo))
similar(v::MPS) = MPS(similar(v.mps),v.dir)
Base.copy(A::MPO) = MPO(copy(A.mpo))
Base.copy(v::MPS) = MPS(copy(v.mps),v.dir)
convert(::Type{MPS},tensor::Array{Array{Complex128,3},1}) = MPS(tensor,:ket)
convert(::Type{Array{Array{Complex128,3},1}},mps::MPS) = mps.mps
promote_rule(::Type{MPS}, ::Type{Array{Array{Complex128,3},1}}) = MPS
MPS(v::Array{Array{Complex{Float64},3},1}) = MPS(v,:ket)

function useMPOasMPS(mpo::MPO,f)
    L = length(mpo)
    smps = Array{Any}(L)
    mps = Array{Array{Complex128,3},1}(L)
    mpoout = similar(mpo)
    for i = 1:L
        smps[i] = size(mpo[i])
        mps[i] = reshape(mpo[i], smps[i][1],smps[i][2]*smps[i][3],smps[i][4])
    end
    mps = f(MPS(mps))
    for i = 1:L
        smps2 = size(mps[i])
        mpoout[i] = reshape(mps[i], smps2[1],smps[i][2],smps[i][3],smps2[3])
    end
    return mpoout
end

function *(A::MPO, v::MPS)
    b = similar(v.mps)
    L = length(v)
    for i = 1:L
        @tensor tmp[:] := A[i][-1,-3,1,-4]*v[i][-2,1,-5]
        s = size(tmp)
        b[i] = reshape(tmp[:],s[1]*s[2],s[3],s[4]*s[5])
    end
    return MPS(b)
end

*(A::MPO, v::Array{Array{Complex{Float64},3},1}) = A*MPS(v)

function *(v1::MPS, v2::MPS)
    if v1.dir==:bra && v2.dir==:ket
        F = Array{Complex128}(1,1)
        F[1,1] = 1
        for i=1:length(v1)
            @tensor F[:] := F[1,2]*v1[i][1,3,-1]*v2[i][2,3,-2]
        end
        return F[1,1]
    elseif v2.dir==:bra && v1.dir==:ket
        return nothing
    elseif v1.dir==:ket && v2.dir==:ket
        return nothing
    elseif v1.dir==:bra && v2.dir==:bra
        return nothing
    end
    return nothing
end

function MPO(L,d,D)
    ### L: length of mps = number of sites/tensors
    ### d: physical dim
    ### D: bond dim
    mpo = Array{Array{Complex128,4}}(L)
    for i = 1:L
        DL = i==1 ? 1 : D
        DR = i==L ? 1 : D
        ran = rand(Complex128,DL*d,d*DR)
        ran = ran/norm(ran)
        mpo[i] = reshape(ran,DL,d,d,DR)
    end
    return MPO(mpo)
end

function MPS(L,d,D)
    ### L: length of mps = number of sites/tensors
    ### d: physical dim
    ### D: bond dim
    mps = Array{Array{Complex128,3}}(L)
    # F[1,1] = 1
    for i = 1:L
        DL = i==1 ? 1 : D
        DR = i==L ? 1 : D
        ran = rand(Complex128,d*max(DL,DR),d*max(DL,DR))
        ran = expm(ran-ran')
        mps[i] = reshape(ran[1:DL,1:DR*d],DL,d,DR)
        # println(mps[i])
        # @tensor F[:] := F[1,2]*conj(mps[i][1,3,-1])*mps[i][2,3,-2]
    end
    # mps = makeCanonical(mps,L)
    mps = makeCanonical(mps,0)
    return MPS(mps,:ket)
end

function *(A::MPO, B::MPO)
    C = copy(A.mpo)
    L = length(A)
    for i = 1:L
        @tensor tmp[:] := A[i][-1,-3,1,-5]*B[i][-2,1,-4,-6]
        s = size(tmp)
        C[i] = reshape(tmp[:],s[1]*s[2],s[3],s[4],s[5]*s[6])
    end
    return MPO(C)
end

function split(block::Array{T,4},Dmax=Inf, dir=:right) where T<:Number
    s = size(block)
    U,S,V = svd(reshape(block,s[1]*s[2],s[3]*s[4]))
    DS = length(S) # number of singular values
    D = Int(min(Dmax,DS))
    if dir == :right
        V = diagm(S[1:D])*V[:,1:D]'
        U = U[:,1:D]
    elseif dir==:left
        U = U[:,1:D]*diagm(S[1:D])
        V = V[:,1:D]'
    end
    Tl = reshape(U,s[1],s[2],D)
    Tr = reshape(V, D,s[3],s[4])
    return Tl,Tr
end

function Base.reduce(v::MPS, Dmax::Int)
    Tlist = makeCanonical(copy(v.mps))
    L = length(Tlist)
    for i=1:L-1
        @tensor block[:] := Tlist[i][-1,-2,1]*Tlist[i+1][1,-3,-4]
        Tlist[i], Tlist[i+1] = split(block, Dmax, :right)
    end
    @tensor block[:] := Tlist[L][-1,-2,1]*ones(1,1,1)[1,-3,-4]
    Tlist[L],_ = split(block, Dmax, :right)
    return MPS(Tlist,v.dir)
end

Base.reduce(A::MPO, Dmax::Int) = useMPOasMPS(A,(v)->reduce(v,Dmax))


function ctranspose(A::MPO)
    B = similar(A.mpo)
    L = length(A)
    for i = 1:L
        @tensor B[i][:] := conj(A[i][-1,-3,-2,-4])
    end
    return MPO(B)
end
function transpose(A::MPO)
    B = similar(A.mpo)
    L = length(A)
    for i = 1:L
        @tensor B[i][:] := A[i][-1,-3,-2,-4]
    end
    return MPO(B)
end

function Base.trace(mpo::MPO)
    L = length(mpo)
    F = Array{Complex128}(1)
    F[1] = 1
    for i = 1:L
        @tensor F[-1] := F[1]*mpo[i][1,2,2,-1]
    end
    return F[1]
end
function norm(mpo::MPO)
    L = length(mpo)
    F = Array{Complex128}(1,1)
    F[1,1] = 1
    for i = 1:L
        @tensor F[-1,-2] := F[1,2]*mpo[i][2,5,4,-2]*conj(mpo[i][1,5,4,-1])
    end
    return F[1,1]
end

function norm(mps::MPS)
    L = length(mps)
    F = Array{Complex128}(1,1)
    F[1,1] = 1
    for i = 1:L
        @tensor F[-1,-2] := F[1,2]*conj(mps[i][1,3,-1])*mps[i][2,3,-2]
    end
    return F[1,1]
end

function norm(mps::Array{Array{T,3},1})  where T<:Number
    L = length(mps)
    F = Array{T}(1,1)
    F[1,1] = 1
    for i = 1:L
        @tensor F[-1,-2] := F[1,2]*conj(mps[i][1,3,-1])*mps[i][2,3,-2]
    end
    return F[1,1]
end

function ctranspose(v::MPS)
    w = similar(v.mps)
    L = length(v)
    for i = 1:L
        @tensor w[i][:] := conj(v[i][-1,-2,-3])
    end
    dir = v.dir==:ket ? :bra : :ket
    return MPS(w,dir)
end

function transpose(v::MPS)
    w = similar(v.mps)
    L = length(v)
    for i = 1:L
        @tensor w[i][:] := v[i][-1,-2,-3]
    end
    dir = v.dir==:ket ? :bra : :ket
    return MPS(w,dir)
end

""" Make mps L-can left of site n and R-can right of site.
    No site specified implies right canonical

    ``` makeCanonical(mps,n=0)```"""
function makeCanonical(mpsin::Array{Array{T,3},1},n=0) where T<:Number
    mps = copy(mpsin)
    L = length(mps)
    for i = 1:n-1
        mps[i],R,DB = LRcanonical(mps[i],-1);
        if i<L
            @tensor mps[i+1][-1,-2,-3] := R[-1,1]*mps[i+1][1,-2,-3];
        end
    end
    for i = L:-1:n+1
        mps[i],R,DB = LRcanonical(mps[i],1);
        if i>1
            @tensor mps[i-1][:] := mps[i-1][-1,-2,1]*R[1,-3]
        end
    end
    return mps
end

makeCanonical(mps::MPS,n=0) = MPS(makeCanonical(mps.mps,n),mps.dir)
makeCanonical(mpo::MPO,n=0) = useMPOasMPS(mpo,(mps)->makeCanonical(mps,n))
    # L = length(mpo)
    # smps = Array{Any}(L)
    # mps = Array{Any}(L)
    # mpoout = similar(mpo)
    # for i = 1:L
    #     smps[i] = size(mpo[i])
    #     mps[i] = reshape(mpo[i], smps[i][1],smps[i][2]*smps[i][3],smps[i][4])
    # end
    # mps = makeCanonical(mps)
    # for i = 1:L
    #     mpoout[i] = reshape(mps[i], smps[i][1],smps[i][2],smps[i][3],smps[i][4])
    # end
    # return mpoout

function LRcanonical(M,dir)
    D1,d,D2 = size(M); # d = phys. dim; D1,D2 = bond dims
    if dir == -1
        M = permutedims(M,[2,1,3]);
        M = reshape(M,D1*d,D2);
        A,R = qr(M); # M = Q R
        DB = size(R)[1]; # intermediate bond dimension
        A = reshape(A,d,D1,DB);
        A = permutedims(A,[2,1,3]);
    elseif dir == 1
        M = permutedims(M,[1,3,2]);
        M = reshape(M,D1,d*D2);
        A,R = qr(M');
        A = A';
        R = R';
        DB = size(R)[2];
        A = reshape(A,DB,D2,d);
        A = permutedims(A,[1,3,2]);
    else println("ERROR: not left or right canonical!");
        A = R = DB = 0;
    end
    return A,R,DB
end
